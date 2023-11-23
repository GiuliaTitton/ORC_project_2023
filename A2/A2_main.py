# -*- coding: utf-8 -*-
import os
import random
import numpy as np

from numpy.linalg import norm
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut

import A2_conf as conf
from single_shooting_problem import SingleShootingProblem

from orc.A2.ode import ODEPendulum
from orc.A2.cost_functions import OCPFinalCostState, OCPRunningCostQuadraticControl
from orc.A2.inequality_constraints import OCPJointPathBounds, OCPJointFinalBounds, OCPVFinalBounds

np.set_printoptions(precision=3, linewidth=200, suppress=True)

PLOT_STUFF = 0

dt = conf.dt                 
T = conf.T
N = int(T/dt);                  # horizon size

nq, nv = conf.nq, conf.nv  
n = nq+nv                       # state size
m = conf.nu                     # control size

GRID_NUM = 10

# TODO implement a search strategy to select n_ics initial states to be checked (for example uniform random sampling, grid-based sampling, etc.)
n_ics = GRID_NUM^2          # TODO number of initial states to be checked
x0_arr = np.zeros((n_ics, n))        # TODO matrix of the initial states to be checked (dim: n_ics x n)

possible_q = np.linspace(conf.lowerPositionLimit, conf.upperPositionLimit, num=GRID_NUM)
possible_v = np.linspace(conf.lowerVelocityLimit, conf.upperVelocityLimit, num=GRID_NUM)

j = k = 0
for i in range(n_ics):
    x0_arr[i, :] = np.array([possible_q[j], possible_v[k]])
    k += 1
    if (k == GRID_NUM):
        k = 0
        j += 1

# Initialize viable and non-viable state lists
viable_states = []
no_viable_states = []

for i in range(int(n_ics)):
    x0 = x0_arr[i,:]
    q0 = x0[:nq]

    # compute initial guess for control inputs
    U = np.zeros((N,m))           
    if(conf.INITIAL_GUESS_FILE is None):
        # use u that compensate gravity
        u0 = 9.81*np.sin(x0[0])
        for j in range(N):
            U[j,:] = u0
    else:
        print("Load initial guess from", conf.INITIAL_GUESS_FILE)
        data = np.load(conf.DATA_FOLDER+conf.INITIAL_GUESS_FILE+'.npz') # , q=X[:,:nq], v=X[:,nv:], u=U
        U = data['u']

    # Create an ODE instance for the analysed system
    ode = ODEPendulum('ode')

    # Create a Single Shooting problem
    problem = SingleShootingProblem('ssp', ode, x0, dt, N, conf.integration_scheme)



    ''' Create cost function terms '''
    # Cost on final velocity
    if(conf.weight_eq_dq>0):
        eq_cost_state = OCPFinalCostState("equilibrium cost state", conf.dq_des_final,
                                             conf.weight_eq_dq)
        problem.add_final_cost(eq_cost_state)

    # Control regularization
    if(conf.weight_u>0):
        effort_cost = OCPRunningCostQuadraticControl("joint torques", dt)
        problem.add_running_cost(effort_cost, conf.weight_u)

    ''' Create constraints '''
    if(conf.activate_joint_bounds):
        q_min  = conf.lowerPositionLimit
        q_max  = conf.upperPositionLimit
        dq_max = conf.upperVelocityLimit
        dq_min = conf.lowerVelocityLimit
        path_bounds = OCPJointPathBounds("path joint bounds", nq, nv, q_min, q_max, dq_min, dq_max)
        problem.add_path_ineq(path_bounds)
        final_bounds = OCPJointFinalBounds("final joint bounds", nq, nv, q_min, q_max, dq_min, dq_max)
        problem.add_final_ineq(final_bounds)

    if(conf.activate_equilibrium_ineq):
        final_bounds_v = OCPVFinalBounds("final joint bounds", nq, nv, -conf.eps_thr, conf.eps_thr)
        problem.add_final_ineq(final_bounds_v)

    ''' Solve OCP '''
    r = problem.solve(y0=U.reshape(N*m), u_bounds=[(conf.lowerControlBound, conf.upperControlBound)],use_finite_diff=conf.use_finite_diff, max_iter = conf.max_iter)
    if r.success:
        # Store OCP solution
        X, U = problem.X, problem.U

        print('{} is a viable x0 - final velocity: {:.3f} rad/s'.format(x0, X[-1,1]))
        # Save viable states
        viable_states.append(x0)         # TODO Save viable states

        # SAVE THE RESULTS
        if(not os.path.exists(conf.DATA_FOLDER)) and conf.save_warm_start:
            os.mkdir(conf.DATA_FOLDER)
            np.savez_compressed(conf.DATA_FOLDER+conf.DATA_FILE_NAME, q=X[:,:nq], v=X[:,nv:], u=U)

        # PLOT STUFF
        if(PLOT_STUFF):    
            time_array = np.arange(0.0, (N+1)*conf.dt, conf.dt)[:N+1]
            for i in range(nq):
                (f, ax) = plut.create_empty_figure(nv,1)
                ax.plot(time_array, X[:,i], label='q')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(r'$q_'+str(i)+'$ [rad]')
                leg = ax.legend()
                leg.get_frame().set_alpha(0.5)
                plt.show()

            for i in range(nq):
                (f, ax) = plut.create_empty_figure(nv,1)
                ax.plot(time_array, X[:,nq+i], label='v')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(r'v_'+str(i)+' [rad/s]')
                leg = ax.legend()
                leg.get_frame().set_alpha(0.5)
                plt.show()

            for i in range(nq):
                (f, ax) = plut.create_empty_figure(nv,1)
                ax.plot(time_array[:-1], U[:,i], label=r'$\tau$ '+str(i))
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Torque [Nm]')
                leg = ax.legend()
                leg.get_frame().set_alpha(0.5)
                plt.show()

            for i in range(nq):
                (f, ax) = plut.create_empty_figure(1)
                ax.plot(problem.history.cost)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Cost')
                plt.show()

            for i in range(nq):
                (f, ax) = plut.create_empty_figure(1)
                ax.plot(problem.history.grad)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Gradient norm')
                ax.set_yscale('log')
                plt.show()
    
    else:
        print('{} is a non-viable x0'.format(x0))
        # Save non-viable states
        no_viable_states.append(x0)         # TODO Save non viable states
        
        

viable_states = np.array(viable_states)
no_viable_states = np.array(no_viable_states)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot()
if len(viable_states) != 0:
    ax.scatter(viable_states[:,0], viable_states[:,1], c='r', label='viable')
    ax.legend()
if len(no_viable_states) != 0:
    ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b', label='non-viable')
    ax.legend()
ax.set_xlabel('q [rad]')
ax.set_ylabel('dq [rad/s]')
plt.show()
