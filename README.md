# Repository of the course "Optimization and Learning for Robot Control"

This repository contains the material of the 2 assignments and of the final project of the course "Optimization and Learning for Robot Control" taken in the first semester of A.A. 2023/2024.

## A1

The folder A1 contains the material of the first assignment, which objectives are:
- Implementing different controllers: operational space control and impedance control.
- Comparing the performance of different versions of the impedance controller to stabilize a desired position of the end effector of a UR5 robot with varying Coulomb friction values.
- Comparing the performance of the controllers using different feedback gains and frequency values of the reference trajectory on the UR5 robot.

## A2

The folder A2 contains the material of the second assignment with the following goals:
- implementing and solving a sequence of optimal control problems to determine a control-invariant set for a single pendulum, taking into account state and input bounds;
- discussing about the shape of the computed set;
- discussing the application of the obtained set for solving Model Predictive Control (MPC) problems.

## A3

Lastly, A3 folder contains the final project. It involves solving multiple Optimal Control Problems (OCPs) from various initial states, either chosen randomly or arranged on a grid.
The goal is to train two neural networks: a "critic" to predict the cost V based on the initial state, and an "actor" to approximate the greedy policy with respect to the critic.
Once trained, the critic can be used for direct system control or to compute initial estimates for the OCP solver.
The analysis starts with a simple 1D single integrator system and progresses to a more complex double integrator system.
The subfolder "Saviane-Titton" contains the final version of the project with the working code.
