import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# Carica set di dati
data = np.load('resultsDoubleInt.npz')
x_data = data['x_init']
velocity_data = data['v_init']
x_grid, vel_grid = np.meshgrid(x_data, velocity_data)
states_data = np.stack((x_grid, vel_grid), axis=-1)
states_data = states_data.reshape((-1, 2, 1))
V_data = data['V']
V_data = np.array(V_data).reshape((-1, 1))

states_tensor = tf.convert_to_tensor(states_data, dtype=tf.float32)
V_tensor = tf.convert_to_tensor(V_data, dtype=tf.float32)

# Crea dataset e mescola i dati
dataset = tf.data.Dataset.from_tensor_slices((states_tensor, V_tensor))
dataset_shuffle = dataset.shuffle(buffer_size=150)
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
V_train = []

for state, V in train_dataset:
    states_train.append(state.numpy())
    V_train.append(V.numpy())

states_train = np.array(states_train)
V_train = np.array(V_train)

# Estrai x_val e V_val come numpy array
states_val = []
V_val = []

for state, V in val_dataset:
    states_val.append(state.numpy())
    V_val.append(V.numpy())

states_val = np.array(states_val)
V_val = np.array(V_val)

# Estrai x_test e V_test come numpy array
states_test = []
V_test = []

for state, V in test_dataset:
    states_test.append(state.numpy())
    V_test.append(V.numpy())

states_test = np.array(states_test)
V_test = np.array(V_test)

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
EPOCHS = 1000
BATCH_SIZE = 32 

start_time=time.time()

print("Fitting model...")
history = model.fit(states_train, V_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(states_val, V_val))

print("Making predictions...")
predictions = model.predict(states_test)
print("Predictions:", predictions)

end_time=time.time() 
print(f"Elapsed time={end_time-start_time} seconds ")

prediction_tot_dataset = model.predict(states_data)
predictions_reshaped = prediction_tot_dataset.reshape(x_grid.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, vel_grid, predictions_reshaped, cmap='viridis')
ax.set_xlabel('Initial position')
ax.set_ylabel('Initial velocity')
ax.set_zlabel('Predicted cost')
ax.set_title('Critic predictions')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

V_data_reshaped = V_data.reshape(x_grid.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, vel_grid, V_data_reshaped, cmap='viridis')
ax.set_xlabel('Initial position')
ax.set_ylabel('Initial velocity')
ax.set_zlabel('OCP cost')
ax.set_title('OCP cost')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

#salva su file i valori predetti 
np.savez('PredictionsForActor.npz', prediction_tot_dataset)

# Salva il modello
model.save('model_critic_doubleIntegrator')