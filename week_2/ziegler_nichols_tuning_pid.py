import os

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq
from simulation_and_control import (
    MotorCommands,
    PinWrapper,
    feedback_lin_ctrl,
    pb,
)

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))


def init_sim():
    sim = pb.SimInterface(
        conf_file_name, conf_file_path_ext=cur_dir, use_gui=False
    )  # Initialize simulation interface
    return sim


# single joint tuning
# episode_duration is specified in seconds
def simulate_with_given_pid_values(
    sim_,
    kp,
    joint_id,
    regulation_displacement=0.1,
    episode_duration=10,
    kd=None,
    plot=False,
):

    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    kp_vec = np.array([15.0] * dyn_model.getNumberofActuatedJoints())
    # updating the kp value for the joint we want to tune
    if isinstance(kp, np.ndarray):
        kp_vec = kp
    elif isinstance(kp, float) or isinstance(kp, int):
        kp_vec[joint_id] = kp

    kd_vec = np.array([0.0] * dyn_model.getNumberofActuatedJoints())
    if isinstance(kd, np.ndarray):
        kd_vec = kd
    elif isinstance(kd, float) or isinstance(kd, int):
        kd_vec[joint_id] = kd

    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())

    q_des[joint_id] = q_des[joint_id] + regulation_displacement

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    (q_mes_all, qd_mes_all, q_d_all, qd_d_all) = ([], [], [], [])

    steps = int(episode_duration / time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(
            dyn_model, q_mes, qd_mes, q_des, qd_des, kp_vec, kd_vec
        )  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord("q")
        if (
            qKey in keys
            and keys[qKey]
            and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED
        ):
            break

        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        # cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes, qdd_est)
        # regressor_all = np.vstack((regressor_all, cur_regressor))

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print("current time in seconds", current_time)

    # make the plot for the current joint
    if plot:
        plot_joint_pos(q_mes_all, q_d_all, kp, joint_id)

    return q_mes_all, q_d_all


def perform_frequency_analysis(data, dt, show=True):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[: n // 2]
    power = 2.0 / n * np.abs(yf[: n // 2])
    # Filter out frequencies below 2 Hz
    filtered_indices = xf <= 2.0
    xf = xf[filtered_indices]
    power = power[filtered_indices]

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.suptitle(f"Joint {joint_id} FFT of the signal")
    plt.title(f"Dominant Frequency: {xf[np.argmax(power)]:.1f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig(
            f"./Estimation_and_Control_labs/week_2/figures/joint_{joint_id}_fft.png",
            dpi=300,
        )
    plt.close()

    return xf, power


# Implement the table in the function
def ziegler_nichols_table(Ku, Tu, control_type):
    if control_type == "P":
        return {"Kp": 0.5 * Ku}
    elif control_type == "PI":
        return {"Kp": 0.45 * Ku, "Ti": 0.83 * Tu, "Ki": 0.54 * Ku / Tu}
    elif control_type == "PD":
        return {"Kp": 0.8 * Ku, "Td": 0.125 * Tu, "Kd": 0.10 * Ku * Tu}
    elif control_type == "PID":
        return {
            "Kp": 0.6 * Ku,
            "Ti": 0.5 * Tu,
            "Td": 0.125 * Tu,
            "Ki": 1.2 * Ku / Tu,
            "Kd": 0.075 * Ku * Tu,
        }
    raise NotImplementedError("Control type not implemented")


def plot_joint_pos(
    q,
    q_d,
    kp,
    joint_id,
    concave_down: np.ndarray = None,
    concave_up: np.ndarray = None,
    line_down=None,
    line_up=None,
    show=True,
) -> None:
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    if not isinstance(q_d, np.ndarray):
        q_d = np.array(q_d)

    plt.figure(figsize=(10, 8))
    plt.plot(q)
    plt.plot(q_d, "--")
    plt.grid(True)
    plt.legend(["Measured", "Desired"])
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.suptitle(f"Joint {joint_id}")
    plt.title(f"Kp={kp}")

    if all(
        v is not None for v in [concave_down, concave_up, line_down, line_up]
    ):
        x = np.arange(len(q))
        plt.plot(x, line_down, "--")
        plt.plot(x, line_up, "--")
        plt.plot(concave_down, q[concave_down], "ro", markersize=5)
        plt.plot(concave_up, q[concave_up], "ro", markersize=5)
        plt.legend(
            [
                "Measured",
                "Desired",
                "Upper Regression Line",
                "Lower Regression Line",
                "Inflection Points",
            ]
        )

    if show:
        plt.show()
    else:
        plt.savefig(
            f"./Estimation_and_Control_labs/week_2/figures/joint_{joint_id}_Kp_{kp}.png",
            dpi=300,
        )
    plt.close()


if __name__ == "__main__":
    sim = init_sim()

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(
        np.array(ext_names), axis=0
    )  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(
        conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir
    )
    num_joints = dyn_model.getNumberofActuatedJoints()

    init_joint_angles = sim.GetInitMotorAngles()
    print(f"Initial joint angles: {init_joint_angles}")

    regulation_displacement = [
        1.0,
        -0.25,
    ]  # Displacement from the initial joint position
    test_duration = 10  # in seconds
    Ku = [16.875, 7.5, 12.65625, 11.25, 16.875, 15.9375, 16.875]
    Tu = [
        1.4285714285714284,
        2.5,
        1.6666666666666665,
        2.0,
        1.4285714285714284,
        1.6666666666666665,
        1.4285714285714284,
    ]
    Kp = [13.5, 6.0, 10.125, 9.0, 13.5, 12.75, 13.5]
    Kd = [
        2.4107142857142856,
        1.875,
        2.109375,
        2.25,
        2.4107142857142856,
        2.6562499999999996,
        2.4107142857142856,
    ]
    Ku, Tu, Kp, Kd = [], [], [], []  # comment this line to skip recomputing

    # using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    for joint_id in range(num_joints):  # Joint ID to tune
        lower_kp = 0
        upper_kp = 30
        for displacement in regulation_displacement:
            switch_displacement = False
            if len(Ku) > joint_id:  # skip if already tuned
                break
            print(
                f"[ ] Tuning Joint {joint_id} with displacement={displacement}"
            )

            # binary search for kp
            while upper_kp - lower_kp > 0.001 and not switch_displacement:
                cur_kp = (upper_kp + lower_kp) / 2  # middle gain to test
                q, q_d = simulate_with_given_pid_values(
                    init_sim(),
                    cur_kp,
                    joint_id,
                    regulation_displacement=displacement,
                    episode_duration=test_duration,
                    plot=False,
                )  # q_mes_all, q_d_all
                q = np.vstack(q)[:, joint_id]
                q_d = np.vstack(q_d)[:, joint_id]
                q_des = q_d[0]

                # find inflection points
                concave_down, concave_up = [], []
                # first point
                if q[0] > q[1] and q[0] >= q_des:
                    concave_down.append(0)
                elif q[0] < q[1] and q[0] <= q_des:
                    concave_up.append(0)
                # other points
                for i in range(1, len(q) - 1):
                    if q[i] >= q[i - 1] and q[i] > q[i + 1] and q[i] >= q_des:
                        concave_down.append(i)
                    elif q[i] <= q[i - 1] and q[i] < q[i + 1] and q[i] <= q_des:
                        concave_up.append(i)
                concave_down = np.array(concave_down)
                concave_up = np.array(concave_up)

                # joint could never reach desired position
                if len(concave_down) == 0 or len(concave_up) == 0:
                    print(
                        f"[-] Joint {joint_id} could not reach desired position with displacement={displacement}"
                    )
                    switch_displacement = True  # try a different displacement
                    break

                # linear regression line for inflection points
                x = np.arange(len(q))
                m_down, b_down = np.polyfit(concave_down, q[concave_down], 1)
                m_up, b_up = np.polyfit(concave_up, q[concave_up], 1)
                line_down = m_down * x + b_down
                line_up = m_up * x + b_up

                mse = np.mean(
                    (q[concave_down] - (m_down * concave_down + b_down)) ** 2
                ) + np.mean((q[concave_up] - (m_up * concave_up + b_up)) ** 2)

                m = (m_down - m_up) / 2  # bottom line is flipped
                print(f"Joint {joint_id} Kp={cur_kp} \t mse={mse} \t m={m}")

                # break if good enough (a good fit and a slope close to 0)
                if abs(mse) < 5e-3 and abs(m) < 7e-6:
                    xf, power = perform_frequency_analysis(
                        q - q_d, sim.GetTimeStep(), False
                    )
                    dominant_freq = xf[np.argmax(power)]
                    Ku.append(cur_kp)
                    Tu.append(1 / dominant_freq)
                    Kp.append(ziegler_nichols_table(Ku[-1], Tu[-1], "PD")["Kp"])
                    Kd.append(ziegler_nichols_table(Ku[-1], Tu[-1], "PD")["Kd"])
                    print(f"[+] Joint {joint_id} Ku={cur_kp}")
                    print(f"[+] Joint {joint_id} dom_freq={dominant_freq}")
                    plot_joint_pos(
                        q,
                        q_d,
                        cur_kp,
                        joint_id,
                        concave_down,
                        concave_up,
                        line_down,
                        line_up,
                        show=False,
                    )
                    break

                # update search space based on the slope
                if m > 0:  # Kp is too high
                    upper_kp = cur_kp  # search in the lower half
                else:  # Kp is too low
                    lower_kp = cur_kp  # search in the upper half
                print("-" * 80)
        if any(len(arr) <= joint_id for arr in [Ku, Tu, Kp, Kd]):
            raise ValueError(
                f"Joint {joint_id} could never reach desired position"
            )
        print("=" * 80)

    # print the results
    print(f"Ku = {Ku}")
    print(f"Tu = {Tu}")
    print(f"Kp = {Kp}")
    print(f"Kd = {Kd}")
    print("=" * 80)

    # plot the results
    for i in range(num_joints):
        q, q_d = simulate_with_given_pid_values(
            init_sim(),
            np.array(Kp),
            i,
            regulation_displacement[
                0 if i != 1 else 1
            ],  # Joint 1 has a different displacement
            test_duration,
            kd=np.array(Kd),
        )
        q = np.vstack(q)[:, i]
        q_d = np.vstack(q_d)[:, i]

        plt.figure(figsize=(10, 8))
        plt.plot(q)
        plt.plot(q_d, "--")
        plt.grid(True)
        plt.legend(["Measured", "Desired"])
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.suptitle(f"Joint {i} under PD Control")
        plt.title(f"Kp={Kp[i]} Kd={Kd[i]}")
        plt.savefig(
            f"./Estimation_and_Control_labs/week_2/figures/joint_{i}_pd.png",
            dpi=300,
        )
        plt.close()
