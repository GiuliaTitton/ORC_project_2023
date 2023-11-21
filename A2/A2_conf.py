# -*- coding: utf-8 -*-
import random
import numpy as np

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

system='pendulum'

T = 1.0                         # OCP horizon
dt = 0.01                       # OCP time step
integration_scheme = 'RK-4'
use_finite_diff = 0
max_iter = 100

DATA_FOLDER = 'data/'     # your data folder name
DATA_FILE_NAME = 'warm_start' # your data file name
save_warm_start = 0
use_warm_start = 0
if use_warm_start:
    INITIAL_GUESS_FILE = DATA_FILE_NAME
else:
    INITIAL_GUESS_FILE = None

lowerPositionLimit = 3/4*np.pi      # min joint position
upperPositionLimit = 5/4*np.pi      # max joint position
upperVelocityLimit = 10             # min joint velocity
lowerVelocityLimit = -10            # min joint velocity
lowerControlBound    = -9.81        # lower bound joint torque
upperControlBound    = 9.81         # upper bound joint torque

activate_joint_bounds = 1           # joint pos/vel bounds

activate_equilibrium_ineq = 0       # equilibrium constraint (inequality)
eps_thr = 1e-7                      # threshold for equilibrium constraint (inequality)

weight_eq_dq = 1e-2                    # final cost weight for joint velocities (to ensure an equilibrium is reached) - ex. 1e-2
dq_des_final = np.array([0])        # final desired joint velocity

weight_u   = 1e-8                   # running cost weight for joint torques

nq = 1                              # number of joint position
nv = 1                              # number of joint velocity
nu = 1                              # number of control