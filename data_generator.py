import numpy as np

class DataGenerator:

    def __init__(self, num_robots, bounds, max_velocity, time_step, input_deviation, measurement_deviation):
        self.velocity = np.zeros((3, num_robots))
        self.num_robots = num_robots
        self.bounds = bounds
        self.max_velocity = max_velocity
        self.time_step = time_step
        self.input_deviation = input_deviation
        self.measurement_deviation = measurement_deviation
        self.move_change_time = 100 # [ms]

        self.e_intX = 0
        self.e_prevX = 0
        self.e_intY = 0
        self.e_prevY = 0

    def random_fly_inputs(self, step):
        if step % self.move_change_time == 0:
                self.velocity[0:2,:] = np.random.uniform(0, self.max_velocity * 2, (2, self.num_robots)) - self.max_velocity
                self.velocity[2,:] = np.random.uniform(0, 1, (1, self.num_robots)) - 1.0
        return self.velocity
    
    def PIDControl(self, relative_x, relative_y, target_x_dist, target_y_dist):
        [kp, kd, ki] = [1.5, 0.001, 0.0001]
        e_x = relative_x - target_x_dist
        self.e_intX = self.e_intX + e_x
        self.e_intX = np.clip(-10, 10, self.e_intX)
        control_x = kp*e_x + kd*(e_x-self.e_prevX) + ki*self.e_intX
        self.e_prevX = e_x
        e_y = relative_y - target_y_dist
        self.e_intY = self.e_intY + e_y
        self.intErrY = np.clip(-10, 10, self.e_intY)
        control_y = kp*e_y + kd*(e_y-self.e_prevY) + ki*self.e_intY
        self.e_prevY = e_y
        return control_x, control_y

    def formation_inputs(self, step, relative_states):
        if step % self.move_change_time == 0:
            self.velocity[0:2,:] = np.random.uniform(0, self.max_velocity*2, (2, self.num_robots)) - self.max_velocity
            self.velocity[2,:] = np.random.uniform(0, 1, (1, self.num_robots)) - 1.0
        if step > 10*self.move_change_time:
            self.velocity[2,:] = np.zeros((1, self.num_robots))
            self.velocity[0, 0], self.velocity[1, 0] = self.PIDControl(relative_states[0, 0, 1], relative_states[1, 0, 1], 1, 1)
            self.velocity[2, 0] = 0
        return self.velocity


    def update_states(self, true_states, control_inputs):
        true_states = self.motion_model(true_states, control_inputs)
        distances = np.zeros((self.num_robots, self.num_robots))
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                dx = true_states[0, i] - true_states[0, j]
                dy = true_states[1, i] - true_states[1, j]
                distances[i, j] = np.sqrt(dx**2 + dy**2)
        noisy_distances = distances + np.random.randn(self.num_robots, self.num_robots) * self.measurement_deviation
        noisy_control_inputs = control_inputs + np.random.randn(3, self.num_robots) * self.input_deviation
        return true_states, noisy_distances, noisy_control_inputs

    def motion_model(self, states, controls):
        predicted_states = np.zeros((3, self.num_robots))
        for i in range(self.num_robots):
            rotation_matrix = np.array([[np.cos(states[2, i]), -np.sin(states[2, i]), 0],
                                        [np.sin(states[2, i]), np.cos(states[2, i]), 0],
                                        [0, 0, 1]]) * self.time_step
            predicted_states[:, i] = states[:, i] + rotation_matrix @ controls[:, i]
        return predicted_states