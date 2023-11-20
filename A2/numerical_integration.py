# -*- coding: utf-8 -*-
import numpy as np

class Integrator:
    ''' A class implementing different numerical integrator schemes '''
    def __init__(self, name):
        self.name = name
        
    def integrate(self, ode, x_init, U, t_init, dt, ndt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory:
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - ndt: the number of inner time steps for each time step
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        n = x_init.shape[0]
        t = np.zeros((N*ndt+1))*np.nan
        x = np.zeros((N*ndt+1,n))*np.nan
        dx = np.zeros((N*ndt,n))*np.nan
        h = dt/ndt  # inner time step
        x[0,:] = x_init
        t[0] = t_init
        
        for i in range(x.shape[0]-1):
            ii = int(np.floor(i/ndt))
            t[i+1] = t[i] + h

            if(scheme=='RK-4'):
                x[i+1,:], dx[i,:] = rk4(x[i,:], h, U[ii,:], t[i], ode)
            else:
                print('{} not implemented'.format(scheme))
            
        self.dx = dx
        self.t = t
        self.x = x        
        return x[::ndt,:]
        
        
    def integrate_w_sensitivities_u(self, ode, x_init, U, t_init, dt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory.
            Compute also the derivative of the x trajectory w.r.t. U.
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        nx = x_init.shape[0]
        nu = ode.nu
        t = np.zeros((N+1))*np.nan
        x = np.zeros((N+1,nx))*np.nan
        dx = np.zeros((N+1,nx))*np.nan
        dXdU = np.zeros(((N+1)*nx,N*nu))
        h = dt
        x[0,:] = x_init
        t[0] = t_init
        
        for i in range(N):
            if(scheme=='RK-4'):
                x[i+1,:], dx[i,:], phi_x, phi_u = rk4(x[i,:], h, U[i,:], t[i], ode, True)
            else:
                return None
            
            t[i+1] = t[i] + h
            ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
            iu, iu1 = i*nu, (i+1)*nu
            dXdU[ix1:ix2,:] = phi_x.dot(dXdU[ix:ix1,:]) 
            dXdU[ix1:ix2,iu:iu1] += phi_u

        self.dx = dx
        self.t = t
        self.x = x        
        return (x, dXdU)

def rk4(x, h, u, t, ode, jacobian=False):
    ''' Runge-Kutta 4 integration scheme '''
    if(not jacobian):
        k1 = ode.f(x,            u, t)
        k2 = ode.f(x + 0.5*h*k1, u, t+0.5*h)
        k3 = ode.f(x + 0.5*h*k2, u, t+0.5*h)
        k4 = ode.f(x + h * k3,   u, t+h)
        dx = (k1 + 2*k2 + 2*k3 + k4)/6.0
        x_next = x + h*dx
        return x_next, dx
    nx = x.shape[0]
    I = np.identity(nx)    

    (k1, f1_x, f1_u) = ode.f(x, u, t, jacobian=True)
    k1_x = f1_x
    k1_u = f1_u

    x2 = x + 0.5*h*k1
    t2 = t+0.5*h
    (k2, f2_x, f2_u) = ode.f(x2, u, t2, jacobian=True)
    k2_x = f2_x.dot(I + 0.5*h*k1_x)
    k2_u = f2_u + 0.5*h*f2_x @ k1_u

    x3 = x + 0.5*h*k2
    t3 = t+0.5*h
    (k3, f3_x, f3_u) = ode.f(x3, u, t3, jacobian=True)
    k3_x = f3_x.dot(I + 0.5*h*k2_x)
    k3_u = f3_u + 0.5*h*f3_x @ k2_u

    x4 = x + h * k3
    t4 = t+h
    (k4, f4_x, f4_u) = ode.f(x4, u, t4, jacobian=True)
    k4_x = f4_x.dot(I + h*k3_x)
    k4_u = f4_u + h*f4_x @ k3_u

    dx = (k1 + 2*k2 + 2*k3 + k4)/6.0
    x_next = x + h*dx

    phi_x = I + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
    phi_u =     h*(k1_u + 2*k2_u + 2*k3_u + k4_u)/6.0

    return x_next, dx, phi_x, phi_u