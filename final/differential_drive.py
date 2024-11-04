import numpy as np
import math
import time
from typing import Literal
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel
from robot_localization_system import FilterConfiguration, Map, RobotEstimator

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])


def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y


from scipy.spatial.transform import Rotation

def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()
    #print(rot_quat)

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    #print(base_euler)

    e2 = Rotation.from_quat([q_w, q_x, q_y, q_z], scalar_first=True)
    #print(e2.as_euler('xyz', degrees=True))

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


print(quaternion2bearing(0.9239, 0, 0, 0.3827))

rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)

# Convert to quaternions and print
rot_quat = rot.as_quat()
print(rot_quat)

# Position change:
# [2,   3]    45 deg: [0, 0,  0.3827,     0.9239]
# [-2,  3]   -45 deg: [0, 0, -0.38268343, 0.92387953]  Q = [311, 311, 311]
# [-1, -4]    76 deg: [0, 0,  0.61566148, 0.78801075]  Q = [309, 309, 309]
# [ 3, -2]   -33 deg: [0, 0, -0.28401534, 0.95881973]

# Orientation change:
# [2, 3]   -45 deg: [0, 0, -0.38268343, 0.92387953]
# [2, 3]    90 deg: [0, 0,  0.70710678, 0.70710678]
# [2, 3]     0 deg: [0, 0, 0, 1]

#exit(0)


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


class Observations(object):

    # Initialize
    def __init__(self, filter_config, map):
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
    def set_true_state(self, x_pos):
        self._x_true = x_pos

    # Get the observations to the landmarks. Return None if none visible
    def landmark_range_observations(self):
        y = []
        C = []
        W = self._filter_config.W_range
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W))
            y.append(range_meas)

        y = np.array(y)
        return y

    def landmark_bearing_observations(self):
        b = []
        W = self._filter_config.W_bearing
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W))
            b.append(bearing_meas)
        
        b = np.array(b)
        return b

    def landmark_range_bearing_observations(self):
        r = self.landmark_range_observations()
        b = self.landmark_bearing_observations()
        y = []
        for i in range(len(r)):
            y.append(r[i])
            y.append(b[i])
        y = np.array(y)
        return y

class Mode:
    GT = "ground_truth"
    EKF = "ekf"


def main():
    # MPC and etc: might be tuned
    N_mpc = 8
    Qcoeff = np.array([312, 312, 312])
    Rcoeff = 0.5
    TASK = 3   # 2, 3, 4, TASK 1 is in a separate file

    if TASK == 4:
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.GT)
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.EKF)
    elif TASK == 3:
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.EKF)
    elif TASK == 2:
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.GT, False, 'zero')
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.GT, True, 'zero')
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.GT, False)
        run_sim(N_mpc, Qcoeff, Rcoeff, TASK, Mode.GT, True)
    else:
        print(F'WRONG TASK NUM {TASK}')
            

def run_sim(N_mpc, Qcoeff, Rcoeff, task, mode: Mode, term_m: bool = True, mpc = ''):
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    wheel_radius = 0.11
    wheel_base_width = 0.46
    num_states = 3
    num_controls = 2

    sim.SetFloorFriction(1)
    time_step = sim.GetTimeStep()

    init_pos = np.array(sim.bot[0].base_position[:2])
    init_quat = sim.bot[0].base_orientation
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    x_est = [init_pos[0], init_pos[1], init_base_bearing_]
    u_mpc = np.zeros(num_controls)

    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    regulator.updateSystemMatrices(sim, x_est, u_mpc)
    regulator.setCostMatrices(Qcoeff,Rcoeff)

    # EKF: no need to tune
    filter_config = FilterConfiguration(x_est)
    map = Map()
    estimator = RobotEstimator(filter_config, map)
    estimator.start()
    obs = Observations(filter_config, map)
    
    x_true_history = []
    x_est_history = []
    Sigma_est_history = []
    lv, rv = [], []
    ss_x_error, ss_y_error, ss_ori_error = [], [], []
    current_time = 0

    # Robot initial command
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    while True:
        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()
        

        # Kalman filter prediction
        estimator.set_control_input(u_mpc)
        estimator.predict_to(current_time)

        # Get the measurements from the simulator ###########################################
         # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position[:2]
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        base_pos_bearing_no_noise = np.hstack((base_pos_no_noise, base_bearing_no_noise_))
        base_pos_bearing_no_noise = base_pos_bearing_no_noise.flatten()
        

        # Update the filter with the latest observations
        x_true_history.append(base_pos_bearing_no_noise)
        ss_x_error.append(base_pos_no_noise[0])
        ss_y_error.append(base_pos_no_noise[1])
        ss_ori_error.append(base_bearing_no_noise_)
        obs.set_true_state(base_pos_bearing_no_noise)
        
        #y = obs.landmark_range_observations()
        #y = obs.landmark_bearing_observations()
        y = obs.landmark_range_bearing_observations()
        #estimator.update_from_landmark_range_observations(y)
        #estimator.update_from_landmark_bearing_observations(y)
        estimator.update_from_landmark_range_bearing_observations(y)


        # Get the current state estimate
        x_est, Sigma_est = estimator.estimate()
        x_est_history.append(x_est)
        Sigma_est_history.append(np.diagonal(Sigma_est))


        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        if mpc != 'zero':
            regulator.updateSystemMatrices(sim, base_pos_bearing_no_noise, u_mpc)


        # Compute the matrices needed for MPC optimization
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std_choice(with_term_m=term_m)
        # [Alternative] compute the optimal control sequence using terminal cost
        #S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std_with_term_cost()
        
        # Compute the optimal control sequence
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        H_inv = np.linalg.inv(H)
        if mode == Mode.EKF:
            u_mpc = -H_inv @ F @ x_est  # WITH EKF
        else:
            u_mpc = -H_inv @ F @ base_pos_bearing_no_noise  # WITHOUT EKF (ground truth)
        u_mpc = u_mpc[0:num_controls]
        #print(u_mpc)


        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


        lv.append(left_wheel_velocity)
        rv.append(right_wheel_velocity)

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Update current time
        current_time += time_step
        if current_time > 10:
            break

    x_true_history = np.array(x_true_history)
    x_est_history = np.array(x_est_history)
    Sigma_est_history = np.array(Sigma_est_history)

    # Plotting the true path, estimated path, and landmarks.
    plt.figure()
    plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path')
    plt.scatter(init_pos[0], init_pos[1], color='green', s=20, label='Init position')
    if mode == Mode.EKF:
        plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path', linestyle='dashed')
    plt.scatter(map.landmarks[:, 0], map.landmarks[:, 1],
                marker='x', color='red', label='Landmarks')
    plt.legend()
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    if mode == Mode.EKF:
        plt.title('Unicycle Robot Localization using EKF')
    else:
        plt.title('Unicycle Robot Localization using ground truth measurements')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    if mode == Mode.EKF:
        plt.savefig(f'task{task}/{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_ekf.pdf')
    elif mpc == 'zero':
        plt.savefig(f'task{task}/{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_zerolin_gt.pdf')
    else:
        plt.savefig(f'task{task}/{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_gt.pdf')


    def get_overshoots(p, t):
        o = []
        eps = 0.0001
        f = 0
        for i in range(1, len(p)):
            if p[0] > 0:
                if p[i] < -eps and not f:
                    o.append(i)
                    f = 1
                elif p[i] > eps:
                    f = 0
            else:
                if p[i] > eps and not f:
                    o.append(i)
                    f = 1
                elif p[i] < -eps:
                    f = 0
        return np.array(o)

    # Plotting the steady state errors and settling time.
    plt.figure()
    err = [ss_x_error, ss_y_error, ss_ori_error]
    FF = ['X', 'Y', 'θ']

    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(err[i])
        tolerance_band = abs(err[i][0]) * 0.07  # 7% tolerance
        settling_time = np.max(np.where(np.abs(err[i]) > tolerance_band)[0])
        overshoots = get_overshoots(err[i], tolerance_band)
        plt.axhline(y=-tolerance_band, linestyle='--', linewidth=0.03)
        plt.axhline(y=tolerance_band, linestyle='--', linewidth=0.03)
        if settling_time != len(err[i]) - 1:
            plt.axvline(x=settling_time, color='g', linestyle='--', label='7% Settling Time')
        for j, o in enumerate(overshoots):
            if j > 0:
                plt.axvline(x=o, color='r', linestyle='--')
            else:
                plt.axvline(x=o, color='r', linestyle='--', label='Overshoot')
        plt.xlabel('Time steps')
        plt.ylabel('Error')
        plt.title(f'Steady state {FF[i]} response')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

    if mode == 'ekf':
        plt.savefig(f'task{task}/sse_{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_ekf.pdf')
    elif mpc == 'zero':
        plt.savefig(f'task{task}/sse_{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_zerolin_gt.pdf')
    else:
        plt.savefig(f'task{task}/sse_{init_pos[0]}_{init_pos[1]}_{math.degrees(init_base_bearing_):.0f}_{term_m}_gt.pdf')

    # plt.figure()
    # plt.plot(lv, label='Left wheel velocity')
    # plt.plot(rv, label='Right wheel velocity')   
    # plt.title('Velocities') 
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # def wrap_angle(angle): return np.arctan2(np.sin(angle), np.cos(angle))

    # # Plot the 2 standard deviation and error history for each state.
    # state_name = ['x', 'y', 'θ']
    # estimation_error = x_est_history - x_true_history
    # estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])
    # for s in range(3):
    #     plt.figure()
    #     two_sigma = 2*np.sqrt(Sigma_est_history[:, s])
    #     plt.plot(estimation_error[:, s])
    #     plt.plot(two_sigma, linestyle='dashed', color='red')
    #     plt.plot(-two_sigma, linestyle='dashed', color='red')
    #     plt.title(state_name[s])
    #     plt.show()
    
    
    

if __name__ == '__main__':
    main()