#!/usr/bin/env python3

import numpy as np


def wrap_angle(angle):
    """Note the angle state theta experiences "angles wrapping". This small helper function is used to address the issue."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range and bearing measurements)
        self.W_range = 0.5**2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2
        self.W_range_bearing = np.diag([self.W_range, self.W_bearing])

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class Map(object):
    def __init__(self):
        self.landmarks = np.array([[5, 10], [15, 5], [10, 15]])

    def use_more_landmarks(self, n):
        """Apply a grid of n landmarks centered at the origin"""
        n = int(np.sqrt(n))
        self.landmarks = (
            np.array(
                [[i - n // 2, j - n // 2] for i in range(n) for j in range(n)]
            )
            * 25
        )


class RobotEstimator(object):
    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    def start(self):
        """This method MUST be called to start the filter"""
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    def predict_to(self, time):
        """Predict to the time. The time is fed in to allow for variable prediction intervals."""
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    def estimate(self):
        """Return the estimate and its covariance"""
        return self._x_est, self._Sigma_est

    def copy_prediction_to_estimate(self):
        """This method gets called if there are no observations"""
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    def _set_estimate_to_initial_conditions(self):
        """This method sets the filter to the initial state"""
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    def _predict_over_dt(self, dt):
        """Predict to the time"""
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array(
            [
                v_c * np.cos(self._x_est[2]) * dt,
                v_c * np.sin(self._x_est[2]) * dt,
                omega_c * dt,
            ]
        )
        self._x_pred[-1] = wrap_angle(self._x_pred[-1])

        # Predict the covariance
        A = np.array(
            [
                [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
                [0, 1, v_c * np.cos(self._x_est[2]) * dt],
                [0, 0, 1],
            ]
        )

        self._kf_predict_covariance(A, self._config.V * dt)

    def _kf_predict_covariance(self, A, V):
        """Predict the EKF covariance; note the mean is totally model specific, so there's nothing we can clearly separate out."""
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    def _do_kf_update(self, nu, C, W):
        """Implement the Kalman filter update step."""
        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):
        # Predicted the landmark measurements and build up the observation Jacobian
        x_pred = self._x_pred
        y_pred = []
        C = []
        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array(
                [-(dx_pred) / range_pred, -(dy_pred) / range_pred, 0]
            )
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are observing a bunch of landmarks build the covariance matrix. Note you could swap this to just calling the ekf update call multiple times, once for each observation, as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = wrap_angle(self._x_est[-1])

    def update_from_landmark_range_bearing_observations(self, y_range_bearing):
        # Predicted the landmark measurements and build up the observation Jacobian
        x_pred = self._x_pred
        y_pred = []
        C = []
        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            y_pred.append([range_pred, bearing_pred])

            # Jacobian of the measurement model
            C_range_bearing = np.array(
                [
                    [-(dx_pred) / range_pred, -(dy_pred) / range_pred, 0],
                    [dy_pred / (range_pred**2), -dx_pred / (range_pred**2), -1],
                ]
            )
            C.append(C_range_bearing)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range_bearing - y_pred
        nu[:, 1] = wrap_angle(nu[:, 1])

        # Since we are observing a bunch of landmarks build the covariance matrix. Note you could swap this to just calling the ekf update call multiple times, once for each observation, as well
        W_landmarks = np.kron(
            np.eye(len(self._map.landmarks)), self._config.W_range_bearing
        )
        self._do_kf_update(nu.flatten(), C.reshape(-1, 3), W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = wrap_angle(self._x_est[-1])
