#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from robot_localization_system import (
    FilterConfiguration,
    Map,
    RobotEstimator,
    wrap_angle,
)


class SimulatorConfiguration(object):
    """Simulator configuration; this only contains stuff relevant for the standalone simulator."""

    def __init__(self):
        self.dt = 0.1
        self.total_time = 1000
        self.time_steps = int(self.total_time / self.dt)

        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]


class Controller(object):
    """Placeholder for a controller."""

    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]


class Simulator(object):
    """This class implements a simple simulator for the unicycle robot seen in the lectures."""

    def __init__(self, sim_config, filter_config, _map):
        """Initialize"""
        self._config = sim_config
        self._filter_config = filter_config
        self._map = _map

    def start(self):
        """Reset the simulator to the start conditions"""
        self._time = 0
        self._x_true = np.random.multivariate_normal(
            self._filter_config.x0, self._filter_config.Sigma0
        )
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    def step(self):
        """Predict the state forwards to the next timestep"""
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        v = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=self._filter_config.V * dt
        )
        self._x_true = (
            self._x_true
            + np.array(
                [
                    v_c * np.cos(self._x_true[2]) * dt,
                    v_c * np.sin(self._x_true[2]) * dt,
                    omega_c * dt,
                ]
            )
            + v
        )
        self._x_true[-1] = np.arctan2(
            np.sin(self._x_true[-1]), np.cos(self._x_true[-1])
        )
        self._time += dt
        return self._time

    def landmark_range_observations(self):
        """Get the observations to the landmarks. Return None if none visible"""
        y = []
        W = self._filter_config.W_range
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W))
            y.append(range_meas)
        return np.array(y)

    def landmark_range_bearing_observations(self):
        """Get the observations to the landmarks and bearings. Return None if none visible"""
        y = []
        W = self._filter_config.W_range_bearing
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W[0, 0]))
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W[1, 1]))
            y.append([range_meas, bearing_meas])
        return np.array(y)

    def x_true(self):
        return self._x_true


TASK = 1
ACT = 3
EXT = "pdf"  # figure extension
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DIR = os.path.join(CUR_DIR, "figures", f"task{TASK}", f"act{ACT}")  # figure dir
os.makedirs(DIR, exist_ok=True)
print(f"Performing Task {TASK} Activity {ACT}...")

# Create the simulator configuration.
sim_config = SimulatorConfiguration()

# Create the filter configuration. If you want to investigate mis-tuning the filter, create a different filter configuration for the simulator and for the filter, and change the parameters between them.
filter_config = FilterConfiguration()

# Create the map object for the landmarks.
_map = Map()
if ACT == 2:
    num_landmarks = 13**2
    _map.use_more_landmarks(num_landmarks)
elif ACT == 4:
    num_landmarks = 7**2
    _map.use_more_landmarks(num_landmarks)

# Create the controller. This just provides fixed control inputs for now.
controller = Controller(sim_config)

# Create the simulator object and start it.
simulator = Simulator(sim_config, filter_config, _map)
simulator.start()

# Create the estimator and start it.
estimator = RobotEstimator(filter_config, _map)
estimator.start()

# Extract the initial estimates from the filter (which are the initial conditions) and use these to generate the control for the first timestep.
x_est, Sigma_est = estimator.estimate()
u = controller.next_control_input(x_est, Sigma_est)

# Arrays to store data for plotting
x_true_history = []
x_est_history = []
Sigma_est_history = []

# Main loop
for step in range(sim_config.time_steps):
    # Set the control input and propagate the step the simulator with that control input.
    simulator.set_control_input(u)
    simulation_time = simulator.step()

    # Predict the Kalman filter with the same control inputs to the same time.
    estimator.set_control_input(u)
    estimator.predict_to(simulation_time)

    # Get the landmark observations.
    # Update the filter with the latest observations.
    if ACT in [1, 2]:
        y = simulator.landmark_range_observations()
        estimator.update_from_landmark_range_observations(y)
    elif ACT in [3, 4]:
        y = simulator.landmark_range_bearing_observations()
        estimator.update_from_landmark_range_bearing_observations(y)

    # Get the current state estimate.
    x_est, Sigma_est = estimator.estimate()

    # Figure out what the controller should do next.
    u = controller.next_control_input(x_est, Sigma_est)

    # Store data for plotting.
    x_true_history.append(simulator.x_true())
    x_est_history.append(x_est)
    Sigma_est_history.append(np.diagonal(Sigma_est))

# Convert history lists to arrays.
x_true_history = np.array(x_true_history[1:])
x_est_history = np.array(x_est_history[1:])
Sigma_est_history = np.array(Sigma_est_history[1:])

# Plotting the true path, estimated path, and landmarks.
plt.figure()
plt.plot(x_true_history[:, 0], x_true_history[:, 1], label="True Path")
plt.plot(x_est_history[:, 0], x_est_history[:, 1], label="Estimated Path")
plt.scatter(
    _map.landmarks[:, 0],
    _map.landmarks[:, 1],
    marker="x",
    color="red",
    label=f"{num_landmarks if ACT in [2, 4] else 3} Landmarks",
)
plt.legend()
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Unicycle Robot Localization using EKF")
plt.axis("equal")
plt.grid(True)
plt.savefig(f"{DIR}/robot_localization.{EXT}")
# plt.show()
plt.close()


# Plot the 2 standard deviation and error history for each state.
state_name = ["x", "y", "θ"]
estimation_error = x_est_history - x_true_history
estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])
for s in range(3):
    plt.figure()
    two_sigma = 2 * np.sqrt(Sigma_est_history[:, s])
    plt.plot(estimation_error[:, s], label="Estimation Error")
    plt.plot(two_sigma, linestyle="dashed", color="r", label="$\pm 2\sigma$")
    plt.plot(-two_sigma, linestyle="dashed", color="r")
    plt.xlabel("Time Step (ms)")
    plt.ylabel(f"Estimation Error ({'m' if state_name[s] != 'θ' else 'rad'})")
    plt.title(state_name[s])
    if state_name[s] in ["x", "y"]:
        plt.hlines(
            [0.10, -0.10],
            0,
            1,
            transform=plt.gca().get_yaxis_transform(),
            color="orange",
            linestyles="dashdot",
            label="10 cm Threshold",
        )
        plt.title(
            f"{state_name[s]} (Threshold{' ' if np.all(two_sigma < 0.10) else ' Not '}Met)"
        )
    plt.legend()
    plt.savefig(f"{DIR}/{state_name[s]}.{EXT}")
    # plt.show()
    plt.close()
