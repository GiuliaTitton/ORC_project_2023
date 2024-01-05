import numpy as np
import casadi

class OcpDoubleIntegrator:

    def __init__(self, dt, w_u, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x_init,v_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()
        self.x = self.opti.variable(N+1,2) #2 states for position and velocity
        self.u = self.opti.variable(N)
        x = self.x
        u = self.u

        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i,:], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i,:], [x_init, v_init])
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:])

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = (x[i,0]-1.9)*(x[i,0]-1.0)*(x[i,0]-0.6)*(x[i,0]+0.5)*(x[i,0]+1.2)*(x[i,0]+2.1)
            if(i<N):
                self.running_costs[i] += self.w_u * u[i]*u[i]
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        for i in range(N):
            #self.opti.subject_to( x[i+1]==x[i] + self.dt*u[i] )
            self.opti.subject_to(x[i+1,0] ==x[i,0] + self.dt * x[i,1] + 0.5 * self.dt**2 * u[i])  # Position update
            self.opti.subject_to(x[i+1,1] == x[i,1] + self.dt * u[i])
        if self.u_min is not None and self.u_max is not None:
            for i in range(N):
                self.opti.subject_to(self.opti.bounded(self.u_min, u[i], self.u_max))
        self.opti.subject_to(x[0, 0] == x_init)
        self.opti.subject_to(x[0, 1] == v_init)

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    #load file with actor results
    data = np.load('ActorResults.npz')
    control_from_actor=data['prediction_tot_dataset']
    states_data=data['states_data']
    
    N = 10          # horizon size
    dt = 0.1        # time step
    w_u = 0.5
    u_min = -5      # min control input
    u_max = 5       # max control input
    N_OCP = 100     # sqrt of number of OCPs (training episodes)
    plot = 1        # plot states-cost

    #compute new OCP
    ocp = OcpDoubleIntegrator(dt, w_u, u_min, u_max)

    # solve OCP starting from different initial states, based on actor predictions
    x_init = np.linspace(-2.2, 2.0, N_OCP) # array of initial states
    v_init = np.linspace(-1.0, 1.0, N_OCP) # array of initial velocities
    V = np.zeros((N_OCP, N_OCP))                    # array of V(x0) for each initial state
    u_optimal = np.zeros((N_OCP, N_OCP))
    for i in range(0, N_OCP):
        for j in range(0, N_OCP):
            sol = ocp.solve(x_init[i],v_init[j], N, None, control_from_actor)
            V[i,j] = sol.value(ocp.cost)
            # , [ocp.x==[x_init[i], v_init[j]]]
            u_optimal[i, j] = sol.value(ocp.u)[0]
            if j%50==0:
                print("OCP number ", i, "\n Initial position: ", sol.value(ocp.x[0,0]), "Initial velocity: ", sol.value(ocp.x[0,1]),"\n Cost: ", V[i,j])
    #print(f"Array of costs from OCP: {V}")
    #print(f"Mean cost of OCP: {np.mean(V)}")
    if plot:

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_init, v_init)
        ax.plot_surface(X, Y, V, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Cost')
        ax.set_title('Cost of the OCPs')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_init, v_init)
        u_opt_reshaped = u_optimal.reshape(X.shape)
        ax.plot_surface(X, Y, u_opt_reshaped, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Optimal control')
        ax.set_title('Controls of OCPs starting from different initial states')
        plt.show()

    # Saving data to a Numpy .npz file
    np.savez('ocpAfterActorFromPredictions.npz', x_init=x_init, v_init=v_init, V=V)
    
    print("INITIAL STATES: ", x_init)
    print("INITIAL VELOCITITES: ", v_init)
    print("COSTS: ", V)
    
    #-------------CASE 2----------------#
    #we start from actor's initial guesses
    V = np.zeros((N_OCP, N_OCP))   # array of V(x0) for each initial state
    u_optimal = np.zeros((N_OCP, N_OCP))
    for i in range(0, N_OCP):
        for j in range(0, N_OCP):
            sol = ocp.solve(x_init[i],v_init[j], N, states_data)
            V[i,j] = sol.value(ocp.cost)
            # , [ocp.x==[x_init[i], v_init[j]]]
            u_optimal[i, j] = sol.value(ocp.u)[0]
            if j%50==0:
                print("OCP number ", i, "\n Initial position: ", sol.value(ocp.x[0,0]), "Initial velocity: ", sol.value(ocp.x[0,1]),"\n Cost: ", V[i,j])
    if plot:

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_init, v_init)
        ax.plot_surface(X, Y, V, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Cost')
        ax.set_title('Cost of the OCPs starting from Actor initial states')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_init, v_init)
        u_opt_reshaped = u_optimal.reshape(X.shape)
        ax.plot_surface(X, Y, u_opt_reshaped, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Optimal control')
        ax.set_title('Controls of OCPs starting from different initial states (from actor guesses)')
        plt.show()
    np.savez('ocpAfterActorFromGuesses.npz', x_init=x_init, v_init=v_init, V=V)
