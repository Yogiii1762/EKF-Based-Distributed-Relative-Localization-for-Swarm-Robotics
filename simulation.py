import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from data_generator import DataGenerator
from ekf import ExtendedKalmanFilter

def convert_to_absolute(robot0_pos, relative_positions, true_states, num_robots):
    abs_positions = np.zeros((3, num_robots))
    for i in range(num_robots):
        abs_positions[0, i] = relative_positions[0, 0, i] * np.cos(true_states[2, 0]) - relative_positions[1, 0, i] * np.sin(true_states[2, 0])
        abs_positions[1, i] = relative_positions[0, 0, i] * np.sin(true_states[2, 0]) + relative_positions[1, 0, i] * np.cos(true_states[2, 0])
        abs_positions[2, i] = relative_positions[2, 0, i]
        abs_positions[:, i] += robot0_pos
    return abs_positions

def compute_relative_states(true_states, num_robots):
    relative_states = np.zeros((3, num_robots))
    x0, y0, yaw0 = true_states[:, 0]
    for i in range(num_robots):
        dx = true_states[0, i] - x0
        dy = true_states[1, i] - y0
        dyaw = true_states[2, i] - yaw0
        relative_states[0, i] = dx * np.cos(yaw0) + dy * np.sin(yaw0)
        relative_states[1, i] = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
        relative_states[2, i] = dyaw
    return relative_states

# Simulation parameters
is_show_sim = True
np.random.seed(5000)
bounds = {"xmin":-5, "xmax":5, "ymin":-5, "ymax":5, "zmin":0, "zmax":5}
robot_count = 10
time_step = 0.01
total_time = 50.0
max_velocity = 1
input_deviation = np.array([[0.2, 0.2, 0.015]]).T
measurement_deviation = 0.05
ekf_interval = 2

# Initialize variables
true_states = np.random.uniform(-3, 3, (3, robot_count))
relative_states = np.zeros((3, robot_count, robot_count))
data_gen = DataGenerator(robot_count, bounds, max_velocity, time_step, input_deviation, measurement_deviation)
ekf = ExtendedKalmanFilter(10, 0.1, 0.25, 0.4, 0.1, robot_count)

def update_frame(step):
    global true_states, relative_states
    control_inputs = data_gen.random_fly_inputs(step)
    # control_inputs = data_gen.formation_inputs(step, relative_states)
    true_states, noisy_measurements, noisy_inputs = data_gen.update_states(true_states, control_inputs)
    if step % ekf_interval == 0:
        relative_states = ekf.update(noisy_inputs, noisy_measurements, relative_states, ekf_interval)
    estimated_positions = convert_to_absolute(true_states[:,0], relative_states, true_states, robot_count)
    actual_points.set_data(true_states[0, :], true_states[1, :])
    estimated_points.set_data(estimated_positions[0, :], estimated_positions[1, :])
    actual_heads.set_data(true_states[0, :] + 0.07*np.cos(true_states[2, :]), true_states[1, :] + 0.07*np.sin(true_states[2, :]))
    estimated_heads.set_data(estimated_positions[0, :] + 0.07*np.cos(estimated_positions[2, :]), estimated_positions[1, :] + 0.07*np.sin(estimated_positions[2, :]))
    circle.center = (true_states[0, 0], true_states[1, 0])
    circle.radius = 0#noisy_measurements[0, 1]
    time_label.set_text(f"t={step * time_step:.2f}s")
    return actual_points, estimated_points,circle, actual_heads, estimated_heads, time_label

if is_show_sim:
    figure, plot = plt.subplots()
    plot.set_aspect('equal')
    plot.set(xlim=(bounds["xmin"], bounds["xmax"]), ylim=(bounds["ymin"], bounds["ymax"]))
    plot.set_xlabel('X (m)')
    plot.set_ylabel('Y (m)')
    plot.set_title('Robot Swarm Simulation')
    actual_points, = plot.plot([], [], 'bo', label='True Position')
    estimated_points, = plot.plot([], [], 'ro', label='Estimated Position')
    actual_heads, = plot.plot([], [], 'go', marker=".")
    estimated_heads, = plot.plot([], [], 'go', marker=".")
    plot.legend()
    circle = plt.Circle((0, 0), 0.1, color='purple', fill=False)
    plot.add_patch(circle)
    time_label = plot.text(0.01, 0.97, '', transform=plot.transAxes)
    animation = anim.FuncAnimation(figure, update_frame, frames=None, interval=10, blit=True)
    plt.show()
else:
    is_plot = False
    range_num = 100
    x_times = []
    y_times = []
    yaw_times = []

    seed_end = 4000
    seed_step = 1000

    for seed in range(range_num):
        np.random.seed((seed+1)*seed_step)
        step = 0
        x_estimates = relative_states[:,0,:]
        x_ground_truth = compute_relative_states(true_states, robot_count)
        plotted_data = np.array([x_estimates[0,1], x_ground_truth[0,1], 
                                x_estimates[1,1], x_ground_truth[1,1], 
                                x_estimates[2,1], x_ground_truth[2,1]])
        x_time, y_time, yaw_time = None, None, None
        x_consecutive, y_consecutive, yaw_consecutive = 0, 0, 0
        while total_time >= time_step * step:
            step += 1
            control_inputs = data_gen.random_fly_inputs(step)
            true_states, noisy_measurements, noisy_inputs = data_gen.update_states(true_states, control_inputs)
            if step % ekf_interval == 0:
                relative_states = ekf.update(noisy_inputs, noisy_measurements, relative_states, ekf_interval)
            x_estimates = relative_states[:,0,:]
            x_ground_truth = compute_relative_states(true_states, robot_count)
            plotted_data = np.vstack([plotted_data, np.array([x_estimates[0,1], x_ground_truth[0,1], 
                                    x_estimates[1,1], x_ground_truth[1,1], 
                                    x_estimates[2,1], x_ground_truth[2,1]])])
            
            # Check if estimates are within 5% of ground truth for the first time
            if ((x_estimates[0,1] - x_ground_truth[0,1]) / x_ground_truth[0,1]) <= 0.02:
                x_consecutive += 1
                if x_consecutive >= 10 and x_time is None:
                    x_time = step * time_step
            else:
                x_consecutive = 0

            if ((x_estimates[1,1] - x_ground_truth[1,1]) / x_ground_truth[1,1]) <= 0.02:
                y_consecutive += 1
                if y_consecutive >= 10 and y_time is None:
                    y_time = step * time_step
            else:
                y_consecutive = 0

            if ((x_estimates[2,1] - x_ground_truth[2,1]) / x_ground_truth[2,1]) <= 0.02:
                yaw_consecutive += 1
                if yaw_consecutive >= 10 and yaw_time is None:
                    yaw_time = step * time_step
            else:
                yaw_consecutive = 0

        x_times.append(x_time)
        y_times.append(y_time)
        yaw_times.append(yaw_time)

        if is_plot:
            plotted_data_array = plotted_data.T
            time_plot = np.arange(0, len(plotted_data_array[0]))/100
            figure, (plot1, plot2, plot3) = plt.subplots(3, sharex=True)
            plt.margins(x=0)
            plot1.plot(time_plot, plotted_data_array[0, :])
            plot1.plot(time_plot, plotted_data_array[1, :])
            plot1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
            plot1.grid(True)
            plot2.plot(time_plot, plotted_data_array[2, :])
            plot2.plot(time_plot, plotted_data_array[3, :])
            plot2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
            plot2.grid(True)
            plot3.plot(time_plot, plotted_data_array[4, :], label='EKF Estimate')
            plot3.plot(time_plot, plotted_data_array[5, :], label='Ground Truth')
            plot3.set_ylabel(r"$\mathrm{\psi_{ij}}$ (rad)", fontsize=12)
            plot3.set_xlabel("Time (s)", fontsize=12)
            plot3.grid(True)
            plot3.legend(loc='upper center', bbox_to_anchor=(0.8, 0.6), shadow=True, ncol=1, fontsize=12)
            figure.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in figure.axes[:-1]], visible=False)
            plt.show()
    
    mean_x_time = np.mean(x_times)
    mean_y_time = np.mean(y_times)
    mean_yaw_time = np.mean(yaw_times)

    var_x_time = np.var(x_times)
    var_y_time = np.var(y_times)
    var_yaw_time = np.var(yaw_times)

    data = {
        'Seed': range(range_num),
        'X Time': x_times,
        'Y Time': y_times,
        'Yaw Time': yaw_times,
    }

    df = pd.DataFrame(data)

    mean_data = {
        'Seed': ['Mean'],
        'X Time': [mean_x_time],
        'Y Time': [mean_y_time],
        'Yaw Time': [mean_yaw_time]
    }

    var_data = {
        'Seed': ['Var'],
        'X Time': [var_x_time],
        'Y Time': [var_y_time],
        'Yaw Time': [var_yaw_time]
    }

    mean_df = pd.DataFrame(mean_data)
    var_df = pd.DataFrame(var_data)

    df = pd.concat([df, mean_df, var_df], ignore_index=True)

    print(df)

