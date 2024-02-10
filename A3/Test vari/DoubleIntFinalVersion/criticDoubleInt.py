import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load data from OCP results
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

# Create and shuffle dataset
dataset = tf.data.Dataset.from_tensor_slices((states_tensor, V_tensor))
dataset_shuffle = dataset.shuffle(buffer_size=10500)
#for element in dataset_shuffle.as_numpy_iterator():
#  print(element)

# Create train, test and validation dataset
train_dataset = dataset_shuffle.take(7000)

start_index_test = 8500
test_dataset = dataset_shuffle.skip(start_index_test).take(1500)

start_index_val = 7000
val_dataset = dataset_shuffle.skip(start_index_val).take(1500)

# Extract x_train and V_train as numpy array
states_train = []
V_train = []

for state, V in train_dataset:
    states_train.append(state.numpy())
    V_train.append(V.numpy())

states_train = np.array(states_train)
V_train = np.array(V_train)

# Extract x_val and V_val as numpy array
states_val = []
V_val = []

for state, V in val_dataset:
    states_val.append(state.numpy())
    V_val.append(V.numpy())

states_val = np.array(states_val)
V_val = np.array(V_val)

# Extract x_test and V_test as numpy array
states_test = []
V_test = []

for state, V in test_dataset:
    states_test.append(state.numpy())
    V_test.append(V.numpy())

states_test = np.array(states_test)
V_test = np.array(V_test)

# Build the model
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

# Train and validate the model 
EPOCHS = 70
BATCH_SIZE = 32 

print("Fitting model...")
history = model.fit(states_train, V_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(states_val, V_val))

# Test the model
print("Making predictions...")
predictions = model.predict(states_test)

# Make predictions on the whole dataset
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

# Save predicted data
np.savez('PredictionsForActor.npz', prediction_tot_dataset)

# Save the model
model.save('model_critic_doubleIntegrator')