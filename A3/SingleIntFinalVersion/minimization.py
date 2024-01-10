import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load x_data
data = np.load('results.npz')
x_data = data['x_init']

# Time step
dt = 0.1
# Joint torque discretization
u_max = 5
n_u = 200
u_vector = np.zeros(n_u+1)
for i in range(n_u+1):
    u_vector[i] = -u_max + i*2*u_max/n_u

# Load critic model (already trained)
model_critic = tf.keras.models.load_model('model_critic')
#model_critic.summary()

# Running cost
def l(x,u):
    x_term = (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1)
    u_term = 0.5 * u**2 
    cost = u_term + x_term 
    return cost

# Create input for actor NN minimizing Q

print("Computing pi...")
pi = np.zeros(len(x_data))
# For each initial state
for i in range(len(x_data)):
    x_next = np.zeros(len(u_vector))
    # Find possible x_next depending on u
    for j in range(len(u_vector)):
        x_next[j] = x_data[i] + dt*u_vector[j]
    # Predict V(x_next) for each possible u
    V_pred_dataset = model_critic.predict(x_next, verbose=0)
    # Transform V_pred from dataset to numpy array
    V_pred = np.array(V_pred_dataset)
    V_pred = V_pred.flatten()
    # Find greedy policy minimizing Q
    Q = np.zeros(len(u_vector))
    for j in range(len(u_vector)):
        Q[j] = l(x_data[i], u_vector[j]) + V_pred[j]
    pi_pos = np.argmin(Q)
    pi[i] = u_vector[pi_pos]

# Plot results 
plt.plot(pi)
plt.xlabel('Prediction number (initial states)')  
plt.ylabel('Policy')  
plt.title('Greedy policy based on critic predictions')     
plt.grid(True)   
plt.show()  

# Save results
np.savez('minimization_results.npz', x_init=x_data, pi=pi)