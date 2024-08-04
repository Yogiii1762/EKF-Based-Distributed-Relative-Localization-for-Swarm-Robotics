import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, init_pos_var, init_yaw_var, proc_pos_var, proc_yaw_var, meas_var, num_robots):
        self.init_pos_var = init_pos_var
        self.init_yaw_var = init_yaw_var
        self.proc_pos_var = proc_pos_var
        self.proc_yaw_var = proc_yaw_var
        self.meas_var = meas_var
        self.num_robots = num_robots
        self.cov_matrix = np.zeros((3, 3, self.num_robots, self.num_robots))
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                self.cov_matrix[0:2, 0:2, i, j] = np.eye(2) * init_pos_var
                self.cov_matrix[2, 2, i, j] = init_yaw_var

    def update(self, noisy_inputs, noisy_measurements, relative_states, interval):
        Q = np.diag([self.proc_pos_var, self.proc_pos_var, self.proc_yaw_var, self.proc_pos_var, self.proc_pos_var, self.proc_yaw_var])**2
        R = np.diag([self.meas_var])**2
        dt = interval * 0.01
        first_in_pair_index = 0
        for j in [k for k in range(self.num_robots) if k != first_in_pair_index]:
            u_i = noisy_inputs[:, first_in_pair_index]
            u_j = noisy_inputs[:, j]
            x_rel, y_rel, yaw_rel = relative_states[:, first_in_pair_index, j]
            state_derivative = np.array([np.cos(yaw_rel) * u_j[0] - np.sin(yaw_rel) * u_j[1] - u_i[0] + u_i[2] * y_rel,
                                            np.sin(yaw_rel) * u_j[0] + np.cos(yaw_rel) * u_j[1] - u_i[1] - u_i[2] * x_rel,
                                            u_j[2] - u_i[2]])
            pred_state = relative_states[:, first_in_pair_index, j] + state_derivative * dt

            F_jacobian = np.array([[1, u_i[2] * dt, (-np.sin(yaw_rel) * u_j[0] - np.cos(yaw_rel) * u_j[1]) * dt],
                                    [-u_i[2] * dt, 1, (np.cos(yaw_rel) * u_j[0] - np.sin(yaw_rel) * u_j[1]) * dt],
                                    [0, 0, 1]])
            B_jacobian = np.array([[-1, 0, y_rel, np.cos(yaw_rel), -np.sin(yaw_rel), 0],
                                    [0, -1, -x_rel, np.sin(yaw_rel), np.cos(yaw_rel), 0],
                                    [0, 0, -1, 0, 0, 1]]) * dt
            pred_cov = F_jacobian @ self.cov_matrix[:, :, first_in_pair_index, j] @ F_jacobian.T + B_jacobian @ Q @ B_jacobian.T
            x_rel, y_rel, yaw_rel = pred_state
            pred_meas = dist = np.sqrt(x_rel**2 + y_rel**2)
            H_jacobian = np.array([[x_rel / dist, y_rel / dist, 0]])
            innovation = noisy_measurements[first_in_pair_index, j] - pred_meas
            S = H_jacobian @ pred_cov @ H_jacobian.T + R
            K = pred_cov @ H_jacobian.T @ np.linalg.inv(S)
            relative_states[:, [first_in_pair_index], [j]] = pred_state.reshape((3, 1)) + K @ np.array([[innovation]])
            self.cov_matrix[:, :, first_in_pair_index, j] = (np.eye(len(pred_state)) - K @ H_jacobian) @ pred_cov
        return relative_states