import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import product

# Load data from minimization results
data = np.load('minimization_results_double.npz')
x_data = data['x_init']
velocity_data = data['v_init']
x_grid, vel_grid = np.meshgrid(x_data, velocity_data)
states = np.stack((x_grid, vel_grid), axis=-1)
states_reshaped = states.reshape((-1, 2, 1))
states_rotated = np.array(list(product(states_reshaped[:, 0, 0], np.unique(states_reshaped[:, 1, 0]))))
states_data_all = states_rotated.reshape((100,-1, 2, 1))
states_data = states_data_all[0,:,:,:]
pi = data['pi']
pi = np.array(pi).reshape((-1, 1))

# Carica critic allenato
model_critic = tf.keras.models.load_model('model_critic_doubleIntegrator')
print("Model critic loaded: ")
model_critic.summary()

# Crea dataset e mescola i dati
states_tensor = tf.convert_to_tensor(states_data, dtype=tf.float32)
pi_tensor = tf.convert_to_tensor(pi, dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((states_tensor, pi_tensor))
dataset_shuffle = dataset.shuffle(buffer_size=10500)

# Crea train, test e validation dataset 
train_dataset = dataset_shuffle.take(7000)

start_index_test = 8500
test_dataset = dataset_shuffle.skip(start_index_test).take(1500)

start_index_val = 7000
val_dataset = dataset_shuffle.skip(start_index_val).take(1500)

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
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1)
])
model.summary()

start_time_train = time.time()

print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',  # Use mean squared error for regression
              metrics=['mse'])  # Use mse for evaluation
print("Model compiled successfully")

# Allena e valuta il modello 
EPOCHS = 15
BATCH_SIZE = 32 

print("Fitting model...")
history = model.fit(states_train, pi_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(states_val, pi_val))

print("Making predictions...")
predictions = model.predict(states_test)
#print("Predictions:", predictions)

end_time_train=time.time()
print(f"Elapsed time in training actor: {end_time_train-start_time_train} seconds")

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

# Save predicted data  
np.savez('PredictionsForOCP_double.npz', prediction_tot_dataset=prediction_tot_dataset)

# Save the model
model.save('model_actor_double')