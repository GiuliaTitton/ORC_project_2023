# -*- coding: utf-8 -*-
import numpy as np
import A2_conf as conf

class OCPJointFinalBounds:
    ''' Final inequality constraint for joint bounds. The constraint is defined as:
            q >= q_min
            -q >= q_max
            dq >= dq_min
            -dq >= dq_max
    '''
    def __init__(self, name, nq, nv, q_min, q_max, dq_min, dq_max):
        self.name = name
        self.nq = nq
        self.nv = nv
        self.q_min = q_min
        self.q_max = q_max
        self.dq_min = dq_min
        self.dq_max = dq_max
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = ineq = np.concatenate((q-self.q_min, self.q_max-q, v-self.dq_min, self.dq_max-v))      # implement the inequality constraint
        return ineq
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = np.concatenate((q-self.q_min, self.q_max-q, v-self.dq_min, self.dq_max-v))  # TODO implement the inequality constraint
        # compute Jacobian
        nx = x.shape[0]
        grad_x = np.zeros((2*nx, nx))
        nq = self.nq
        grad_x[:nq,       :nq] = np.eye(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[nq:2*nq,   :nq] = -np.eye(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[2*nq:3*nq, nq:] = np.eye(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[3*nq:,     nq:] = -np.eye(nq)        # TODO implement the jacobian of the inequality constraint

        return (ineq, grad_x)

class OCPVFinalBounds:
    ''' Final inequality constraint for joint bounds. The constraint is defined as:
            dq >= dq_min
            -dq >= dq_max
    '''
    def __init__(self, name, nq, nv, dq_min, dq_max):
        self.name = name
        self.nq = nq
        self.nv = nv
        self.dq_min = dq_min
        self.dq_max = dq_max
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = np.concatenate((v+conf.eps_thr, conf.eps_thr-v))        # TODO implement the jacobian of the inequality constraint
        return ineq
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = np.concatenate((v+conf.eps_thr, conf.eps_thr-v)) # TODO implement the inequality constraint 
        # compute Jacobian
        nx = x.shape[0]
        grad_x = np.zeros((2*nx, nx))
        nq = self.nq
        grad_x[:nq,       :nq] = np.zeros(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[nq:2*nq,   :nq] = np.zeros(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[2*nq:3*nq, nq:] = np.eye(nq)        # TODO implement the jacobian of the inequality constraint
        grad_x[3*nq:,     nq:] = -np.eye(nq)        # TODO implement the jacobian of the inequality constraint


        return (ineq, grad_x)

class OCPJointPathBounds:
    ''' Path inequality constraint for joint bounds. The constraint is defined as:
            q >= q_min
            -q >= q_max
            dq >= dq_min
            -dq >= dq_max
    '''
    def __init__(self, name, nq, nv, q_min, q_max, dq_min, dq_max):
        self.c = OCPJointFinalBounds(name, nq, nv, q_min, q_max, dq_min, dq_max)
        self.name = name
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost given the state x '''
        return self.c.compute(x, recompute)
    
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        (ineq, grad_x) = self.c.compute_w_gradient(x, recompute)
        grad_u = np.zeros((ineq.shape[0], u.shape[0]))
        return (ineq, grad_x, grad_u)
    