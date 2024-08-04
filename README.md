# EKF-Based Distributed Relative Localization for Swarm Robotics

## Overview

This project implements a distributed relative localization system for a swarm of small drones using Extended Kalman Filter (EKF) techniques. The primary goal is to achieve accurate localization in multi-robot systems operating under resource constraints and high speeds, where traditional GPS or fixed ultra-wideband (UWB) systems may not be suitable.

## Project Objectives

The main objectives of this project are:

- **Accurate Localization:** To maintain precise relative positioning among a swarm of drones in a 2D plane using UWB communication and EKF-based sensor fusion.

- **Scalability and Robustness:** To design a system that can function reliably under various conditions, with emphasis on high-speed operations and significant resource limitations of small drones.

- **Simulation Framework:** To provide a Python-based simulation environment that models the behavior of flying micro-robots, testing the EKF's performance in both random and controlled flight scenarios.

## Implementation Details

The implementation consists of several key components:

- **Communication System:** The drones are equipped with UWB radios that facilitate accurate distance measurements using a round-robin communication model. This enables each drone to estimate its relative position to others by exchanging distance data.

- **Extended Kalman Filter (EKF):** The EKF is used to estimate the relative positions of the drones by fusing data from onboard sensors and UWB measurements. The state and measurement functions are adapted to handle the non-linearities inherent in the system.

- **Simulation Environment:** The project includes a Python simulator that visualizes the swarm behavior and evaluates the performance of the EKF in estimating the drones' positions. The simulator allows for the tuning of Process Noise Covariance and Measurement Noise Covariance matrices for optimal performance.

## Key Features

- **Sensor Fusion:** Integrates data from inertial measurement units (IMU), optical flow sensors, and UWB radios to maintain accurate localization.

- **Scenarios:** Evaluates performance through two scenarios - random movements and controlled formation flights, highlighting the EKF's accuracy and reliability.

- **Visualization:** Utilizes `matplotlib` to display the true and estimated positions of the drones in a simulated environment.

## Results

The simulation results demonstrate the estimator's accuracy, with errors consistently within 2% of the ground truth in both random and controlled scenarios. Despite occasional outliers, the system shows robustness in maintaining accurate relative positions.

## Limitations and Future Work

### Limitations

- **High Variance in Outlier Cases:** Some simulations exhibit significant divergence from the mean, indicating areas for improvement in handling outlier cases.

- **2D Movements:** The current implementation is limited to two-dimensional movements, restricting applicability in scenarios where height (Z-axis) is a critical factor.

### Future Work

- **Reducing Variance:** Focus on refining the algorithm to handle outlier cases more effectively and ensure consistent performance across different scenarios.

- **3D Extension:** Expand the estimator's capabilities to include three-dimensional localization, enabling applications in aerial and underwater robotics.
