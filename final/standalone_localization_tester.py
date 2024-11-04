#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from robot_localization_system import FilterConfiguration, Map, RobotEstimator

# Simulator configuration; this only contains
# stuff relevant for the standalone simulator.


class SimulatorConfiguration(object):
    def __init__(self):
        self.dt = 0.1
        self.total_time = 1000
        self.time_steps = int(self.total_time / self.dt)

        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]

# Placeholder for a controller.


class Controller(object):
    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]

# This class implements a simple simulator for the unicycle
# robot seen in the lectures.


class Simulator(object):

    # Initialize
    def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    # Reset the simulator to the start conditions
    def start(self):
        self._time = 0
        self._x_true = np.random.multivariate_normal(self._filter_config.x0,
                                                     self._filter_config.Sigma0)
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    # Predict the state forwards to the next timestep
    def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        v = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=self._filter_config.V * dt)
        self._x_true = self._x_true + np.array([
            v_c * np.cos(self._x_true[2]) * dt,
            v_c * np.sin(self._x_true[2]) * dt,
            omega_c * dt
        ]) + v
        self._x_true[-1] = np.arctan2(np.sin(self._x_true[-1]),
                                      np.cos(self._x_true[-1]))
        self._time += dt
        return self._time

    # Get the observations to the landmarks. Return None if none visible
    def landmark_range_observations(self):
        y_r = []
        W = self._filter_config.W_range
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W))
            y_r.append(range_meas)

        y_r = np.array(y_r)
        return y_r
    def landmark_bearing_observations(self):
        y_b = []
        W = self._filter_config.W_bearing 

        for lm in self._map.landmarks:
            # True bearing measurement (relative to the robot's orientation)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            bearing_true = np.arctan2(np.sin(bearing_true), np.cos(bearing_true))# Normalize the bearing to be within [-pi, pi]
            bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W))
            bearing_meas = np.arctan2(np.sin(bearing_meas), np.cos(bearing_meas))# Normalize the noisy bearing measurement to be within [-pi, pi]
            y_b.append(bearing_meas)

        y_b = np.array(y_b)
        return y_b

    def x_true(self):
        return self._x_true


# Create the simulator configuration.
sim_config = SimulatorConfiguration()

# Create the filter configuration. If you want
# to investigate mis-tuning the filter,
# create a different filter configuration for
# the simulator and for the filter, and
# change the parameters between them.
filter_config = FilterConfiguration()

# Create the map object for the landmarks.
map = Map()
# Create the controller. This just provides
# fixed control inputs for now.
controller = Controller(sim_config)

# Create the simulator object and start it.
simulator = Simulator(sim_config, filter_config, map)
simulator.start()

# Create the estimator and start it.
estimator = RobotEstimator(filter_config, map)
estimator.start()

# Extract the initial estimates from the filter
# (which are the initial conditions) and use
# these to generate the control for the first timestep.
x_est, Sigma_est = estimator.estimate()
u = controller.next_control_input(x_est, Sigma_est)

# Arrays to store data for plotting
x_true_history = []
x_est_history = []
Sigma_est_history = []

# Main loop
for step in range(sim_config.time_steps):
    # Use some combinations of landmarks
    map.use_more_landmarks()
    # Set the control input and propagate the
    # step the simulator with that control iput.
    simulator.set_control_input(u)
    simulation_time = simulator.step()

    # Predict the Kalman filter with the same
    # control inputs to the same time.
    estimator.set_control_input(u)
    estimator.predict_to(simulation_time)

    # Get the landmark observations.
    y_r = simulator.landmark_range_observations()
    y_b = simulator.landmark_bearing_observations()
    # Update the filter with the latest observations.
    estimator.update_from_landmark_range_observations(y_r)
    estimator.update_from_landmark_bearing_observations(y_b)
    
    # Get the current state estimate.
    x_est, Sigma_est = estimator.estimate()

    # Figure out what the controller should do next.
    u = controller.next_control_input(x_est, Sigma_est)

    # Store data for plotting.
    x_true_history.append(simulator.x_true())
    x_est_history.append(x_est)
    Sigma_est_history.append(np.diagonal(Sigma_est))

# Convert history lists to arrays.
x_true_history = np.array(x_true_history)
x_est_history = np.array(x_est_history)
Sigma_est_history = np.array(Sigma_est_history)

# Plotting the true path, estimated path, and landmarks.
plt.figure()
plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path')
plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path')
plt.scatter(map.landmarks[:, 0], map.landmarks[:, 1],
            marker='x', color='red', label='Landmarks')
plt.legend()
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Unicycle Robot Localization using EKF')
plt.axis('equal')
plt.grid(True)
file_path = f'/Users/huangyuting/Estimation and Control/week_5/figures/task1/activity4_combination_of_landmarks_and_bearing_measurement/activity4_trajectory.png'
try:
    plt.savefig(file_path)
    print(f"Figure for trajectory saved successfully")
except Exception as e:
    print(f"Error saving figure for: {e}")
plt.show()

# Plot the 2 standard deviation and error history for each state.
state_name = ['x', 'y', 'θ']
estimation_error = x_est_history - x_true_history

# Note the angle state theta experiences "angles
# wrapping". This small helper function is used
# to address the issue.
def wrap_angle(angle): return np.arctan2(np.sin(angle), np.cos(angle))

estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])

for s in range(3):
    plt.figure()
    two_sigma = 2*np.sqrt(Sigma_est_history[:, s])
    threshold_pos = np.full_like(two_sigma, 0.1) 
    threshold_neg = np.full_like(two_sigma, -0.1)
    plt.plot(estimation_error[:, s])
    plt.plot(two_sigma, linestyle='dashed', color='red')
    plt.plot(-two_sigma, linestyle='dashed', color='red')
    if s < 2:
        plt.plot(threshold_pos, color='black', linestyle='--', label='10 cm Threshold')
        plt.plot(threshold_neg, color='black', linestyle='--')
    plt.title(state_name[s])
    file_path = f'/Users/huangyuting/Estimation and Control/week_5/figures/task1/activity4_combination_of_landmarks_and_bearing_measurement/activity4_{state_name[s]}_threshold.png'
    try:
        plt.savefig(file_path)
        print(f"Figure for {state_name[s]} saved successfully")
    except Exception as e:
        print(f"Error saving figure for {state_name[s]}: {e}")
    plt.show()
