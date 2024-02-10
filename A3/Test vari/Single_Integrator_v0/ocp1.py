import numpy as np
import casadi

class OcpSingleIntegrator:

    def __init__(self, dt, w_u, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()
        self.x = self.opti.variable(N+1)
        self.u = self.opti.variable(N)
        x = self.x
        u = self.u

        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:])

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = (x[i]-1.9)*(x[i]-1.0)*(x[i]-0.6)*(x[i]+0.5)*(x[i]+1.2)*(x[i]+2.1)
            if(i<N):
                self.running_costs[i] += self.w_u * u[i]*u[i]
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to( x[i+1]==x[i] + self.dt*u[i] )
        if(self.u_min is not None and self.u_max is not None):
            for i in range(N):
                self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max) )
        self.opti.subject_to(x[0]==x_init)

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 10          # horizon size
    dt = 0.1        # time step
    w_u = 0.5
    u_min = -5      # min control input
    u_max = 5       # max control input
    N_OCP = 20000     # number of OCPs (training episodes)
    plot = 1        # plot states-cost


    ocp = OcpSingleIntegrator(dt, w_u, u_min, u_max)

    # solve OCP starting from different initial states
    x_init = np.linspace(-2.2, 2.0, N_OCP) # array of initial states
    V = np.zeros(N_OCP)                    # array of V(x0) for each initial state
    u_optimal = np.zeros(N_OCP)
    for i in range(0, N_OCP):
        sol = ocp.solve(x_init[i], N)
        V[i] = sol.value(ocp.cost, [ocp.x==x_init[i]]) 
        u_optimal[i] = sol.value(ocp.u)[0]
        print("OCP number ", i, "\n Initial state: ", sol.value(ocp.x[0]), "\n Cost: ", V[i], "\n Optimal control: ", u_optimal[i])
    if plot:
        plt.plot(x_init, V)
        plt.xlabel('Initial state')  
        plt.ylabel('Cost')  
        plt.title('Costs of OCPs starting from different initial states')     
        plt.grid(True)  
        plt.show()

        plt.plot(x_init, u_optimal)
        plt.xlabel('Initial state')  
        plt.ylabel('Optimal control')  
        plt.title('Controls of OCPs starting from different initial states')     
        plt.grid(True)  
        plt.show()
    
    #x_init_values = x_init.tolist()
    #V_values = V.tolist()
    
    # Saving data to a Numpy .npz file
    np.savez('results.npz', x_init=x_init, V=V)

    # Loading data from a Numpy .npz file
    data = np.load('results.npz')
    x_init_values = data['x_init']
    V_values = data['V']
    
    print("INITIAL STATES: ", x_init)
    print("COSTS: ", V)
