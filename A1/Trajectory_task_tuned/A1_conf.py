# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

TRACK_TRAJ = 1

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 110

q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ])  # initial configuration
T_SIMULATION = 10             # simulation time
dt = 0.01                   # controller time step
ndt = 1                      # number of integration steps for each control loop

frame_name = 'tool0'    # name of the frame to control (end-effector)

x_ref_O = np.array([0.2,0.2,0.2])

# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY
x0          = np.array([0.632, 0.091, 0.472]).T         # offset
amp         = np.array([0.1, 0.1, 0.0]).T           # amplitude
phi         = np.array([0.0, 0.5*np.pi, 0.0]).T     # phase

tau_coulomb_max = 0
simulation_type = 'timestepping' # either 'timestepping' or 'euler'

randomize_robot_model = 0
model_variation = 40.0

use_viewer = 1
which_viewer = "no_gepetto"
simulate_real_time = 0          # flag specifying whether simulation should be real time or as fast as possible
PRINT_T = 10                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds