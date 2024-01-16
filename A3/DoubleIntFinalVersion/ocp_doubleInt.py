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
            self.opti.subject_to(x[i+1,0] ==x[i,0] + self.dt * x[i,1] + 0.5 * self.dt**2 * u[i]) # Position update
            self.opti.subject_to(x[i+1,1] == x[i,1] + self.dt * u[i])
        if self.u_min is not None and self.u_max is not None:
            for i in range(N):
                self.opti.subject_to(self.opti.bounded(self.u_min, u[i], self.u_max))
        self.opti.subject_to(x[0, 0] == x_init)
        self.opti.subject_to(x[0, 1] == v_init)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 10          # horizon size
    dt = 0.1        # time step
    w_u = 0.5
    u_min = -5      # min control input
    u_max = 5       # max control input
    N_OCP = 100     # sqrt of number of OCPs (training episodes)
    plot = 1        # plot states-cost


    ocp = OcpDoubleIntegrator(dt, w_u, u_min, u_max)

    # solve OCP starting from different initial states
    x_init = np.linspace(-2.2, 2.0, N_OCP) # array of initial states
    v_init = np.linspace(-1.0, 1.0, N_OCP) # array of initial velocities
    V = np.zeros((N_OCP, N_OCP))    # array of V(x0) for each initial state
    u_optimal = np.zeros((N_OCP, N_OCP, N))
    x_traj = np.zeros((N_OCP, N_OCP, N+1, 2))
    for i in range(0, N_OCP):
        for j in range(0, N_OCP):
            sol = ocp.solve(x_init[i],v_init[j], N)
            V[i,j] = sol.value(ocp.cost)
            u_optimal[i,j,:] = sol.value(ocp.u)
            x_traj[i,j,:,:] = sol.value(ocp.x)
            print("OCP number ", i*N_OCP+j, "\n Initial position: ", sol.value(ocp.x[0,0]), "Initial velocity: ", sol.value(ocp.x[0,1]), "\n Cost: ", V[i,j], "Optimal control: ", u_optimal[i,j,:])
    if plot:

        # 3D plots
        # Cost
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_init, v_init)
        ax.plot_surface(X, Y, V, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Cost')
        ax.set_title('Cost of OCPs starting from different initial states')
        plt.show()

        # Plot control
        for k in range(N):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_init, v_init)
            U = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    U[i,j] = u_optimal[i,j,k]
            ax.plot_surface(X, Y, U, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Control')
            ax.set_title('Controls of OCPs starting from different initial states')
            plt.show()

        # Plot states
        # Position
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_init, v_init)
            pos_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    pos_plot[i,j] = x_traj[i,j,k,0]
            ax.plot_surface(X, Y, pos_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Position')
            ax.set_title('Position trajectory of OCPs starting from different initial states')
            plt.show()
        # Velocity
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_init, v_init)
            vel_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    vel_plot[i,j] = x_traj[i,j,k,1]
            ax.plot_surface(X, Y, vel_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Velocity')
            ax.set_title('Velocity trajectory of OCPs starting from different initial states')
            plt.show()

        # Plot trajectory of state (25,25)

        # Position
        plt.plot(x_traj[25,25,:,0])
        plt.xlabel('Time step')  
        plt.ylabel('Position')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()
        # Velocity
        plt.plot(x_traj[25,25,:,1])
        plt.xlabel('Time step')  
        plt.ylabel('Velocity')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()

        # Plot trajectory of state (45,45)

        # Position
        plt.plot(x_traj[45,45,:,0])
        plt.xlabel('Time step')  
        plt.ylabel('Position')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()
        # Velocity
        plt.plot(x_traj[45,45,:,1])
        plt.xlabel('Time step')  
        plt.ylabel('Velocity')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()  

        # Plot trajectory of state (0,0)   
        # Position
        plt.plot(x_traj[0,0,:,0])
        plt.xlabel('Time step')  
        plt.ylabel('Position')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()
        # Velocity
        plt.plot(x_traj[0,0,:,1])
        plt.xlabel('Time step')  
        plt.ylabel('Velocity')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()  
                            
            
    # Saving data to a Numpy .npz file
    np.savez('resultsDoubleInt.npz', x_init=x_init, v_init=v_init, V=V)
    
    print("INITIAL STATES: ", x_init)
    print("INITIAL VELOCITITES: ", v_init)
    print("COSTS: ", V)
