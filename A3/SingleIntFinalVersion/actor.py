import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from minimization results
data = np.load('minimization_results.npz')
x_data = data['x_init']
pi = data['pi']

# Create and shuffle dataset
dataset = tf.data.Dataset.from_tensor_slices((x_data, pi))
dataset_shuffle = dataset.shuffle(buffer_size=20000)
#for element in dataset_shuffle.as_numpy_iterator():
#  print(element)

# Create train, test and validation dataset 
train_dataset = dataset_shuffle.take(14000)

start_index_test = 17000
test_dataset = dataset_shuffle.skip(start_index_test).take(3000)

start_index_val = 14000
val_dataset = dataset_shuffle.skip(start_index_val).take(3000)

# Extract x_train and V_train as numpy array
x_train = []
pi_train = []

for x, pi in train_dataset:
    x_train.append(x.numpy())
    pi_train.append(pi.numpy())

x_train = np.array(x_train)
pi_train = np.array(pi_train)

# Extract x_val and V_val as numpy array
x_val = []
pi_val = []

for x, pi in val_dataset:
    x_val.append(x.numpy())
    pi_val.append(pi.numpy())

x_val = np.array(x_val)
pi_val = np.array(pi_val)

# Extract x_test and V_test as numpy array
x_test = []
pi_test = []

for x, pi in test_dataset:
    x_test.append(x.numpy())
    pi_test.append(pi.numpy())

x_test = np.array(x_test)
pi_test = np.array(pi_test)

nx = 1
# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(nx, 1)),
  tf.keras.layers.Dense(64, activation='relu'),
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
EPOCHS = 60
BATCH_SIZE = 32

print("Fitting model...")
history = model.fit(x_train, pi_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, pi_val))

# Test the model
print("Making predictions...")
predictions = model.predict(x_test)
print("Predictions:", predictions)

# Make predictions on the whole dataset
prediction_tot_dataset = model.predict(x_data)
plt.plot(prediction_tot_dataset)
plt.xlabel('Prediction number (initial states)')  
plt.ylabel('Predicted value (greedy policy)')  
plt.title('Predictions on the whole dataset')     
plt.grid(True)   
plt.show()  

# Save predicted data 
np.savez('PredictionsForOCP.npz', prediction_tot_dataset=prediction_tot_dataset)

# Save the model
model.save('model_actor')