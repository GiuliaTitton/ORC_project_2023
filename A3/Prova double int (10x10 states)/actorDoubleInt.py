import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carica states_data
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

# Carica critic allenato
model_critic = tf.keras.models.load_model('model_critic_doubleIntegrator')
print("Model critic loaded: ")
model_critic.summary()

# Running cost
def l(x,u):
    x_term = (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1)
    u_term = 0.5 * u**2 
    cost = u_term + x_term 
    return cost

# Crea input per l'actor minimizzando Q

pi = np.zeros(len(states_data))
print("Computing pi ...")
# For each initial state
for i in range(len(states_data)):
    x_next = np.zeros(len(u_vector))
    v_next = np.zeros(len(u_vector))
    # Find possible x_next, v_next depending on u
    for j in range(len(u_vector)):
        x_next[j] = states_data[i, 0, 0] + dt*states_data[i, 1, 0] + 0.5 * (dt**2) * u_vector[j]
        v_next[j] = states_data[i, 1, 0] + dt*u_vector[j]
    # Predict V(x_next) for each possible u
    x_grid_next, vel_grid_next = np.meshgrid(x_next, v_next)
    states_next = np.stack((x_grid_next, vel_grid_next), axis=-1)
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

# Crea dataset e mescola i dati
states_tensor = tf.convert_to_tensor(states_data, dtype=tf.float32)
pi_tensor = tf.convert_to_tensor(pi, dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((states_tensor, pi_tensor))
dataset_shuffle = dataset.shuffle(buffer_size=500)
#for element in dataset_shuffle.as_numpy_iterator():
#  print(element)

# Crea train, test e validation dataset 
train_dataset = dataset_shuffle.take(70)

start_index_test = 85
test_dataset = dataset_shuffle.skip(start_index_test).take(15)

start_index_val = 70
val_dataset = dataset_shuffle.skip(start_index_val).take(15)

# Estrai x_train e V_train come numpy array
states_train = []
pi_train = []

for state, pi in train_dataset:
    states_train.append(state.numpy())
    pi_train.append(pi.numpy())

states_train = np.array(states_train)
pi_train = np.array(pi_train)

# Estrai x_val e V_val come numpy array
states_val = []
pi_val = []

for state, pi in val_dataset:
    states_val.append(state.numpy())
    pi_val.append(pi.numpy())

states_val = np.array(states_val)
pi_val = np.array(pi_val)

# Estrai x_test e V_test come numpy array
states_test = []
pi_test = []

for state, pi in test_dataset:
    states_test.append(state.numpy())
    pi_test.append(pi.numpy())

states_test = np.array(states_test)
pi_test = np.array(pi_test)

# Costruisci il modello
nx = 2
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(nx, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])
model.summary()


print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',  # Use mean squared error for regression
              metrics=['mse'])  # Use mse for evaluation
print("Model compiled successfully")

# Allena e valuta il modello 
EPOCHS = 200
BATCH_SIZE = 32 

print("Shape of states_train:", states_train.shape)
print("Shape of pi_train:", pi_train.shape)

print(states_train)
print(pi_train)

print("Fitting model...")
history = model.fit(states_train, pi_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(states_val, pi_val))

print("Making predictions...")
predictions = model.predict(states_test)
print("Predictions:", predictions)

prediction_tot_dataset = model.predict(states_data)
predictions_reshaped = prediction_tot_dataset.reshape(x_grid.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, vel_grid, predictions_reshaped, cmap='viridis')
ax.set_xlabel('Initial position')
ax.set_ylabel('Initial velocity')
ax.set_zlabel('Predicted control')
ax.set_title('Actor predictions')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
num_x, num_vel = x_grid.shape
x_grid_reshaped = x_grid.reshape((num_x * num_vel, 1))
vel_grid_reshaped = vel_grid.reshape((num_x * num_vel, 1))
pi_tensor_reshaped = pi_tensor.numpy().reshape((num_x * num_vel, 1))
print(x_grid_reshaped)
print(vel_grid_reshaped)
print(pi_tensor_reshaped)
surf = ax.plot_surface(x_grid_reshaped, vel_grid_reshaped, pi_tensor_reshaped, cmap='viridis')
ax.set_xlabel('Initial position')
ax.set_ylabel('Initial velocity')
ax.set_zlabel('Optimal control')
ax.set_title('Control based on critic predictions')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

#salva su file i valori predetti 
np.savez('PredictionsForActor.npz', prediction_tot_dataset)