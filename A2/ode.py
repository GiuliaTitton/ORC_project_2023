# -*- coding: utf-8 -*-
import numpy as np

class ODE:
    def __init__(self, name):
        self.name = name
        self.nu = 1
        
    def f(self, x, u, t):
        return np.zeros(x.shape)
             
class ODEPendulum(ODE):
    def __init__(self, name=''):
        ODE.__init__(self, name) 
        self.g = -9.81
        
    def f(self, x, u, t, jacobian=False):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = self.g*np.sin(x[0]) + u

        if(jacobian):
            df_dx =         # TODO implement the Jacobian of the dynamics w.r.t. x
            df_du =         # TODO implement the Jacobian of the dynamics w.r.t. u
            return (dx, df_dx, df_du)
        
        return dx