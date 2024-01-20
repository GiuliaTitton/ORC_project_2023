import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# Load data from OCP results
data = np.load('results.npz')
x_data = data['x_init']
V_data = data['V']

# Create and shuffle dataset
dataset = tf.data.Dataset.from_tensor_slices((x_data, V_data))
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
V_train = []

for x, V in train_dataset:
    x_train.append(x.numpy())
    V_train.append(V.numpy())

x_train = np.array(x_train)
V_train = np.array(V_train)

# Extract x_val and V_val as numpy array
x_val = []
V_val = []

for x, V in val_dataset:
    x_val.append(x.numpy())
    V_val.append(V.numpy())

x_val = np.array(x_val)
V_val = np.array(V_val)

# Extract x_test and V_test as numpy array
x_test = []
V_test = []

for x, V in test_dataset:
    x_test.append(x.numpy())
    V_test.append(V.numpy())

x_test = np.array(x_test)
V_test = np.array(V_test)

nx = 1
# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(nx, 1)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1)
])
model.summary()

start_time = time.time()
print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',  # Use mean squared error for regression
              metrics=['mse'])  # Use mse for evaluation
print("Model compiled successfully")

# Train and validate the model 
EPOCHS = 45
BATCH_SIZE = 32

print("Fitting model...")
history = model.fit(x_train, V_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, V_val))

# Test the model
print("Making predictions...")
predictions = model.predict(x_test)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for training critic: {elapsed_time}")

# Make predictions on the whole dataset
prediction_tot_dataset = model.predict(x_data)
plt.plot(prediction_tot_dataset)
plt.xlabel('Prediction number (initial states)')  
plt.ylabel('Predicted value (cost)')  
plt.title('Predictions on the whole dataset')     
plt.grid(True)   
plt.show()  

# Save predicted data
np.savez('PredictionsForActor.npz', prediction_tot_dataset)

# Save the model
model.save('model_critic')