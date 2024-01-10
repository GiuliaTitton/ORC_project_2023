import numpy as np
import casadi
import tensorflow as tf

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
                self.opti.set_initial(x[i], X_guess[i])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i])

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
    N_OCP = 2000     # number of OCPs 
    plot = 1        # plot states-cost

    # Load actor model
    model_actor = tf.keras.models.load_model('model_actor')

    # Load actor predictions
    data = np.load('PredictionsForOCP.npz')
    control_from_actor = data['prediction_tot_dataset']
    controls = np.array(control_from_actor).flatten()

    # Initial states
    x_init = np.linspace(-2.2, 2.0, N_OCP) # array of initial states
    
    # ------------CASE 1-------------
    # System directly controlled by actor predictions
    
    # Calculate states and controls based on actor predictions
    X = np.zeros((len(x_init), N+1))
    U = np.zeros((len(x_init), N))
    V_direct = np.zeros(len(x_init))
    for i in range(len(x_init)):
        X[i,0] = x_init[i] # initial state
        U[i,0] = controls[int(i*20000/N_OCP)] # first control to be applied (*20000/N_OCP because controls contains 20000 values coming from actor)
        u = controls[int(i*20000/N_OCP)]
        for j in range(N):
            V_direct[i] = V_direct[i] + (X[i,j]-1.9)*(X[i,j]-1.0)*(X[i,j]-0.6)*(X[i,j]+0.5)*(X[i,j]+1.2)*(X[i,j]+2.1) + 0.5*U[i,j]*U[i,j]
            # Compute next state based on actor prediction on u
            x_next = X[i,j] + dt*u
            X[i,j+1] = x_next
            # Predict next control to be applied
            if j<N-1:
                u_pred = model_actor.predict(x_next.reshape(1, 1), verbose=0)
                u = np.array(u_pred)
                U[i,j+1] = u
        
        V_direct[i] = V_direct[i] + (X[i,N]-1.9)*(X[i,N]-1.0)*(X[i,N]-0.6)*(X[i,N]+0.5)*(X[i,N]+1.2)*(X[i,N]+2.1) 

        if i%100==0:
            print(i*100/N_OCP, "%")

    if plot:

        # Plot cost 
        plt.plot(x_init, V_direct)
        plt.xlabel('Initial state')  
        plt.ylabel('Cost')  
        plt.title('Cost with direct control from actor')     
        plt.grid(True)  
        plt.show()

        # Plot trajectory for all the states (from time 0 to N)
        plt.plot(x_init, X[:,0])
        plt.xlabel('Initial state')  
        plt.ylabel('State')  
        plt.title('States of OCPs directly controlled by actor')     
        plt.grid(True)  
        plt.show()
        for i in range(N):
            plt.plot(x_init, X[:,i+1])
            plt.xlabel('Initial state')  
            plt.ylabel('State')  
            plt.title('States of OCPs directly controlled by actor')     
            plt.grid(True)  
            plt.show()

            # Plot control for all the states (from time 0 to N)
            plt.plot(x_init, U[:,i])
            plt.xlabel('Initial state')  
            plt.ylabel('Initial control')  
            plt.title('Controls of OCPs directly controlled by actor')     
            plt.grid(True)  
            plt.show()

    # ------------CASE 2-------------
    # OCP solution with actor predictions as initial guess (both states and control)

    ocp = OcpSingleIntegrator(dt, w_u, u_min, u_max)

    # solve OCP starting from different initial states
    V = np.zeros(N_OCP)  # array of V(x0) for each initial state
    u_optimal = np.zeros((N_OCP, N))
    x_traj = np.zeros((N_OCP, N+1))
    for i in range(0, N_OCP):
        sol = ocp.solve(x_init[i], N, X_guess=X[i,:], U_guess=U[i,:])
        V[i] = sol.value(ocp.cost, [ocp.x==x_init[i]]) 
        u_optimal[i,:] = sol.value(ocp.u)
        x_traj[i,:] = sol.value(ocp.x)
        print("OCP number ", i, "\n Initial state: ", sol.value(ocp.x[0]), "\n Cost: ", V[i], "\n Optimal control: ", u_optimal[i,:])
    
    if plot:

        # Plot cost
        plt.plot(x_init, V)
        plt.xlabel('Initial state')  
        plt.ylabel('Cost')  
        plt.title('Cost of OCPs with initial guess from actor')     
        plt.grid(True)  
        plt.show()    

        # Plot for all states from t = 0 to N
        plt.plot(x_init, x_traj[:,0])
        plt.xlabel('Initial state')  
        plt.ylabel('State')  
        plt.title('States of OCPs with initial guess from actor')     
        plt.grid(True)  
        plt.show()
        for i in range(N):
            
            # Plot states
            plt.plot(x_init, x_traj[:,i+1])
            plt.xlabel('Initial state')  
            plt.ylabel('State')  
            plt.title('States of OCPs with initial guess from actor')     
            plt.grid(True)  
            plt.show()

            # Plot control
            plt.plot(x_init, u_optimal[:,i])
            plt.xlabel('Initial state')  
            plt.ylabel('Optimal control')  
            plt.title('Controls of OCPs with initial guess from actor')     
            plt.grid(True)  
            plt.show()
  
    