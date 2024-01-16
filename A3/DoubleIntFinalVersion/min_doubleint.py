import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load states_data
data = np.load('resultsDoubleInt.npz')
x_data = data['x_init']
velocity_data = data['v_init']
x_grid, vel_grid = np.meshgrid(x_data, velocity_data)
states_data = np.stack((x_grid, vel_grid), axis=-1)
states_data = states_data.reshape((-1, 2, 1))

# Time step
dt = 0.1
# Joint torque discretization
u_max = 5
n_u = 200
u_vector = np.zeros(n_u+1)
for i in range(n_u+1):
    u_vector[i] = -u_max + i*2*u_max/n_u

# Load critic model (already trained)
model_critic = tf.keras.models.load_model('model_critic_doubleIntegrator')
#model_critic.summary()

# Running cost
def l(x,u):
    x_term = (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1)
    u_term = 0.5 * u**2 
    cost = u_term + x_term 
    return cost

# Create input for actor NN minimizing Q

print("Computing pi...")
pi = np.zeros((len(states_data)))
# For each initial state
for i in range(len(states_data)):
    x_next = np.zeros(len(u_vector))
    v_next = np.zeros(len(u_vector))
    # Find possible x_next, v_next depending on u
    for j in range(len(u_vector)):
        x_next[j] = states_data[i, 0, 0] + dt * states_data[i, 1, 0] + 0.5 * (dt**2) * u_vector[j]
        v_next[j] = states_data[i, 1, 0] + dt * u_vector[j]
    # Predict V(states_next) for each possible u
    states_next = np.column_stack((x_next, v_next))
    states_next = states_next.reshape((-1, 2, 1))
    V_pred_dataset = model_critic.predict(states_next, verbose=0)
    # Transform V_pred from dataset to numpy array
    V_pred = np.array(V_pred_dataset)
    V_pred = V_pred.flatten()

    # Find greedy policy minimizing Q
    Q = np.zeros(len(u_vector))
    for j in range(len(u_vector)):
        Q[j] = l(states_data[i, 0, 0], u_vector[j]) + V_pred[j]
    pi_pos = np.argmin(Q)
    pi[i] = u_vector[pi_pos]
    if i%500==0:
        print(i*100/len(states_data), "%")

# Plot results 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_data, velocity_data)
pi_plot = pi.reshape(X.shape)
ax.plot_surface(X, Y, pi_plot, cmap='viridis')
ax.set_xlabel('Initial position')
ax.set_ylabel('Initial velocity')
ax.set_zlabel('Policy')
ax.set_title('Greedy policy based on critic predictions')
plt.show() 

# Save results
np.savez('minimization_results_double.npz', x_init=x_data, v_init=velocity_data, pi=pi)