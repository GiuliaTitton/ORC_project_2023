import numpy as np
import casadi
import tensorflow as tf
import matplotlib.pyplot as plt

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
                self.opti.set_initial(u[i], U_guess[i])

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

    # Initial states
    x_init = np.linspace(-2.2, 2.0, N_OCP) # array of initial states
    v_init = np.linspace(-1.0, 1.0, N_OCP) # array of initial velocities
    x_grid, vel_grid = np.meshgrid(x_init, v_init)

    # Load actor model
    model_actor = tf.keras.models.load_model('model_actor_double')


    # ------------CASE 1-------------
    # System directly controlled by actor predictions
    
    # First step
    X = np.zeros((N_OCP,N_OCP,N+1))
    V = np.zeros((N_OCP,N_OCP,N+1))
    U = np.zeros((N_OCP,N_OCP,N))
    V_direct = np.zeros((N_OCP, N_OCP))
    for i in range(N_OCP):
        for j in range(N_OCP):
            X[i,j,0] = x_init[i]
            V[i,j,0] = v_init[j]

    states = np.dstack((X[:,:,0], V[:,:,0]))
    states = states.reshape((-1, 2, 1))
    pi_predicted = model_actor.predict(states)
    pi_pred = np.array(pi_predicted)
    pi = pi_pred.reshape(x_grid.shape)
    for i in range(N_OCP):
        for j in range(N_OCP):
            U[i,j,0] = pi[i,j]
            V_direct[i,j] = V_direct[i,j] + (X[i,j,0]-1.9)*(X[i,j,0]-1.0)*(X[i,j,0]-0.6)*(X[i,j,0]+0.5)*(X[i,j,0]+1.2)*(X[i,j,0]+2.1) + 0.5*U[i,j,0]*U[i,j,0]
    

    # Next steps
    for k in range(1,N+1):
        for i in range(N_OCP):
            for j in range(N_OCP):
                X[i,j,k] = X[i,j,k-1] + dt*V[i,j,k-1] + 0.5 * dt**2 * U[i,j,k-1]
                V[i,j,k] = V[i,j,k-1] + dt*U[i,j,k-1]

        if k<N:
            states = np.dstack((X[:,:,k], V[:,:,k]))
            states = states.reshape((-1, 2, 1))
            pi_predicted = model_actor.predict(states)
            pi_pred = np.array(pi_predicted)
            pi = pi_pred.reshape(x_grid.shape)
            for i in range(N_OCP):
                for j in range(N_OCP):
                    U[i,j,k] = pi[i,j]
                    V_direct[i,j] = V_direct[i,j] + (X[i,j,k]-1.9)*(X[i,j,k]-1.0)*(X[i,j,k]-0.6)*(X[i,j,k]+0.5)*(X[i,j,k]+1.2)*(X[i,j,k]+2.1) + 0.5*U[i,j,k]*U[i,j,k]
    # Add terminal cost
    for i in range(N_OCP):
        for j in range(N_OCP):
            V_direct[i,j] = V_direct[i,j] + (X[i,j,N]-1.9)*(X[i,j,N]-1.0)*(X[i,j,N]-0.6)*(X[i,j,N]+0.5)*(X[i,j,N]+1.2)*(X[i,j,N]+2.1)


    if plot:

        # 3D plots
        # Cost
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X_plot, Y_plot = np.meshgrid(x_init, v_init)
        ax.plot_surface(X_plot, Y_plot, V_direct, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Cost')
        ax.set_title('Cost with direct control from actor')
        plt.show()

        # Plot control
        for k in range(N):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            U_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    U_plot[i,j] = U[i,j,k]
            ax.plot_surface(X_plot, Y_plot, U_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Control')
            ax.set_title('Controls of OCPs directly controlled by actor')
            plt.show()

        # Plot states
        # Position
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            pos_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    pos_plot[i,j] = X[i,j,k]
            ax.plot_surface(X_plot, Y_plot, pos_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Position')
            ax.set_title('Position trajectory of OCPs directly controlled by actor')
            plt.show()
        # Velocity
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            vel_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    vel_plot[i,j] = V[i,j,k]
            ax.plot_surface(X_plot, Y_plot, vel_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Velocity')
            ax.set_title('Velocity trajectory of OCPs directly controlled by actor')
            plt.show()

        '''# Plot trajectory of state A

        # Position
        plt.plot(X[25,25,:])
        plt.xlabel('Time step')  
        plt.ylabel('Position')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()
        # Velocity
        plt.plot(V[25,25,:])
        plt.xlabel('Time step')  
        plt.ylabel('Velocity')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()

        # Plot trajectory of state B

        # Position
        plt.plot(X[45,45,:])
        plt.xlabel('Time step')  
        plt.ylabel('Position')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()
        # Velocity
        plt.plot(V[45,45,:])
        plt.xlabel('Time step')  
        plt.ylabel('Velocity')  
        plt.title('State trajectory')     
        plt.grid(True)  
        plt.show()     ''' 


    # ------------CASE 2-------------
    # OCP solution with actor predictions as initial guess (both states and control)
    
    ocp = OcpDoubleIntegrator(dt, w_u, u_min, u_max)

    # solve OCP starting from different initial states
    V_cost = np.zeros((N_OCP, N_OCP))    # array of V(x0) for each initial state
    u_optimal = np.zeros((N_OCP, N_OCP, N))
    x_traj = np.zeros((N_OCP, N_OCP, N+1, 2))
    guess_states = np.zeros((N_OCP, N_OCP, N+1, 2))
    guess_states[:,:,:,0] = X
    guess_states[:,:,:,1] = V
    
    for i in range(0, N_OCP):
        for j in range(0, N_OCP):
            sol = ocp.solve(x_init[i], v_init[j], N, X_guess=guess_states[i,j,:,:], U_guess=U[i,j,:])
            V_cost[i,j] = sol.value(ocp.cost)
            u_optimal[i,j,:] = sol.value(ocp.u)
            x_traj[i,j,:,:] = sol.value(ocp.x)
            print("OCP number ", i*N_OCP+j, "\n Initial position: ", sol.value(ocp.x[0,0]), "Initial velocity: ", sol.value(ocp.x[0,1]), "\n Cost: ", V_cost[i,j], "Optimal control: ", u_optimal[i,j,:])
    if plot:

        # 3D plots
        # Cost
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X_plot, Y_plot = np.meshgrid(x_init, v_init)
        ax.plot_surface(X_plot, Y_plot, V_cost, cmap='viridis')
        ax.set_xlabel('Initial position')
        ax.set_ylabel('Initial velocity')
        ax.set_zlabel('Cost')
        ax.set_title('Cost of OCPs with initial guess from actor')
        plt.show()

        # Plot control
        for k in range(N):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            U = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    U[i,j] = u_optimal[i,j,k]
            ax.plot_surface(X_plot, Y_plot, U, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Control')
            ax.set_title('Controls of OCPs with initial guess from actor')
            plt.show()

        # Plot states
        # Position
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            pos_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    pos_plot[i,j] = x_traj[i,j,k,0]
            ax.plot_surface(X_plot, Y_plot, pos_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Position')
            ax.set_title('Position trajectory of OCPs with initial guess from actor')
            plt.show()
        # Velocity
        for k in range(N+1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X_plot, Y_plot = np.meshgrid(x_init, v_init)
            vel_plot = np.zeros((N_OCP,N_OCP))
            for i in range(N_OCP):
                for j in range(N_OCP):
                    vel_plot[i,j] = x_traj[i,j,k,1]
            ax.plot_surface(X_plot, Y_plot, vel_plot, cmap='viridis')
            ax.set_xlabel('Initial position')
            ax.set_ylabel('Initial velocity')
            ax.set_zlabel('Velocity')
            ax.set_title('Velocity trajectory of OCPs with initial guess from actor')
            plt.show()

        ''' # Plot trajectory of state A

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

        # Plot trajectory of state B

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
        plt.show()     '''  