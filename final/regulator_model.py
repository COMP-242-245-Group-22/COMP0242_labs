import numpy as np
from scipy.linalg import solve_discrete_are


class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.N = N
        self.q = q  #  output dimension
        self.m = m  #  input dimension
        self.n = n  #  state dimension

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = S_bar.T @ (Q_bar @ S_bar) + R_bar

        # Compute F
        F = S_bar.T @ (Q_bar @ T_bar)
        return H, F

    def propagation_model_regulator_fixed_std(self, use_term_cost=False):
        if use_term_cost:
            Q_ = self.Q + 1e-8 * np.eye(self.Q.shape[0])
            R_ = self.R + 1e-8 * np.eye(self.R.shape[0])
            P = solve_discrete_are(self.A, self.B, Q_, R_)

        N_ = self.N + use_term_cost
        Nq = N_ * self.q
        Nm = N_ * self.m
        S_bar = np.zeros((Nq, Nm))
        T_bar = np.zeros((Nq, self.n))
        Q_bar = np.zeros((Nq, Nq))
        R_bar = np.zeros((Nm, Nm))

        for k in range(1, N_ + 1):
            k1q, kq = (k - 1) * self.q, k * self.q
            k1m, km = (k - 1) * self.m, k * self.m
            for j in range(1, k + 1):
                kjm, kj1m = (k - j) * self.m, (k - j + 1) * self.m
                S_bar[k1q:kq, kjm:kj1m] = (
                    self.C @ np.linalg.matrix_power(self.A, j - 1)
                ) @ self.B

            T_bar[k1q:kq, :] = self.C @ np.linalg.matrix_power(self.A, k)

            if k == self.N + 1:
                Q_bar[k1q:kq, k1q:kq] = P
                R_bar[k1m:km, k1m:km] = np.zeros((self.m, self.m))
            else:
                Q_bar[k1q:kq, k1q:kq] = self.Q
                R_bar[k1m:km, k1m:km] = self.R

        return S_bar, T_bar, Q_bar, R_bar

    def propagation_model_regulator_fixed_std_with_term_cost(self):
        Q_ = self.Q + 1e-8 * np.eye(self.Q.shape[0])
        R_ = self.R + 1e-8 * np.eye(self.R.shape[0])
        P = solve_discrete_are(self.A, self.B, Q_, R_)
        self.N += 1
        S_bar = np.zeros((self.N * self.q, self.N * self.m))
        T_bar = np.zeros((self.N * self.q, self.n))
        Q_bar = np.zeros((self.N * self.q, self.N * self.q))
        R_bar = np.zeros((self.N * self.m, self.N * self.m))
        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[
                    (k - 1) * self.q : k * self.q,
                    (k - j) * self.m : (k - j + 1) * self.m,
                ] = np.dot(
                    np.dot(self.C, np.linalg.matrix_power(self.A, j - 1)),
                    self.B,
                )
            T_bar[(k - 1) * self.q : k * self.q, : self.n] = np.dot(
                self.C, np.linalg.matrix_power(self.A, k)
            )
            if k == self.N:
                Q_bar[
                    (k - 1) * self.q : k * self.q, (k - 1) * self.q : k * self.q
                ] = P
                R_bar[
                    (k - 1) * self.m : k * self.m, (k - 1) * self.m : k * self.m
                ] = np.zeros((self.m, self.m))
            else:
                Q_bar[
                    (k - 1) * self.q : k * self.q, (k - 1) * self.q : k * self.q
                ] = self.Q
                R_bar[
                    (k - 1) * self.m : k * self.m, (k - 1) * self.m : k * self.m
                ] = self.R
        self.N -= 1
        return S_bar, T_bar, Q_bar, R_bar

    def updateSystemMatrices(self, sim, cur_x, cur_u):
        """
        Get the system matrices A and B according to the dimensions of the state and control input.

        Parameters:
        sim: Simulation object containing the time step information.
        cur_x: Current state around which to linearize.
        cur_u: Current control input around which to linearize.

        Raises:
        ValueError: If cur_x or cur_u is not provided.

        Sets:
        self.A: State transition matrix.
        self.B: Control input matrix.
        """
        # Check if state_x_for_linearization and cur_u_for_linearization are provided
        if cur_x is None or cur_u is None:
            raise ValueError(
                "state_x_for_linearization and cur_u_for_linearization are not specified.\n"
                "Please provide the current state and control input for linearization.\n"
                "Hint: Use the goal state (e.g., zeros) and zero control input at the beginning.\n"
                "Also, ensure that you implement the linearization logic in the updateSystemMatrices function."
            )

        num_outputs = self.q
        delta_t = sim.GetTimeStep()
        v0 = cur_u[0]
        theta0 = cur_x[2]

        A = np.array(
            [
                [1, 0, -v0 * delta_t * np.sin(theta0)],
                [0, 1, v0 * delta_t * np.cos(theta0)],
                [0, 0, 1],
            ]
        )

        B = np.array(
            [
                [delta_t * np.cos(theta0), 0],
                [delta_t * np.sin(theta0), 0],
                [0, delta_t],
            ]
        )

        # Updating the state and control input matrices
        self.A = A
        self.B = B
        self.C = np.eye(num_outputs)

    def setCostMatrices(self, Qcoeff, Rcoeff):
        """
        Set the cost matrices Q and R for the MPC controller.

        Parameters:
        Qcoeff: float or array-like
            State cost coefficient(s). If scalar, the same weight is applied to all states.
            If array-like, should have a length equal to the number of states.
        Rcoeff: float or array-like
            Control input cost coefficient(s). If scalar, the same weight is applied to all control inputs.
            If array-like, should have a length equal to the number of control inputs.

        Sets:
        self.Q: ndarray
            State cost matrix.
        self.R: ndarray
            Control input cost matrix.
        """

        num_states = self.n
        num_controls = self.m

        # Process Qcoeff
        if np.isscalar(Qcoeff):
            # If Qcoeff is a scalar, create an identity matrix scaled by Qcoeff
            Q = Qcoeff * np.eye(num_states)
        else:
            # Convert Qcoeff to a numpy array
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(
                    f"Qcoeff must be a scalar or a 1D array of length {num_states}"
                )
            # Create a diagonal matrix with Qcoeff as the diagonal elements
            Q = np.diag(Qcoeff)

        # Process Rcoeff
        if np.isscalar(Rcoeff):
            # If Rcoeff is a scalar, create an identity matrix scaled by Rcoeff
            R = Rcoeff * np.eye(num_controls)
        else:
            # Convert Rcoeff to a numpy array
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(
                    f"Rcoeff must be a scalar or a 1D array of length {num_controls}"
                )
            # Create a diagonal matrix with Rcoeff as the diagonal elements
            R = np.diag(Rcoeff)

        # Assign the matrices to the object's attributes
        self.Q = Q
        self.R = R
