import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Carica set di dati
data = np.load('resultsDoubleInt.npz')
x_data = data['x_init']
velocity_data = data['v_init']
V_data = data['V']

# Crea dataset e mescola i dati
dataset = tf.data.Dataset.from_tensor_slices((x_data, velocity_data, V_data))
dataset_shuffle = dataset.shuffle(buffer_size=20000)
#for element in dataset_shuffle.as_numpy_iterator():
#  print(element)

# Crea train, test e validation dataset 
train_dataset = dataset_shuffle.take(14000)

start_index_test = 17000
test_dataset = dataset_shuffle.skip(start_index_test).take(3000)

start_index_val = 14000
val_dataset = dataset_shuffle.skip(start_index_val).take(3000)

# Estrai x_train e V_train come numpy array
x_train = []
velocity_train = []
V_train = []

for x, velocity, V in train_dataset:
    x_train.append(x.numpy())
    velocity_train.append(velocity.numpy())
    V_train.append(V.numpy())

x_train = np.array(x_train)
velocity_train = np.array(velocity_train)
V_train = np.array(V_train)

# Estrai x_val e V_val come numpy array
x_val = []
velocity_val = []
V_val = []

for x, vel, V in val_dataset:
    x_val.append(x.numpy())
    velocity_val.append(vel.numpy())
    V_val.append(V.numpy())

x_val = np.array(x_val)
velocity_val = np.array(velocity_val)
V_val = np.array(V_val)

# Estrai x_test e V_test come numpy array
x_test = []
velocity_test = []
V_test = []

for x, vel, V in test_dataset:
    x_test.append(x.numpy())
    velocity_test.append(vel.numpy())
    V_test.append(V.numpy())

x_test = np.array(x_test)
velocity_test = np.array(velocity_test)
V_test = np.array(V_test)

nx = 2
'''# Costruisci modello di apprendimento automatico
model = tf.keras.models.Sequential([
  tf.keras.layers.Concatenate(input_shape=(nx, 1)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])
model.summary()'''

# Define input layers
input_x = tf.keras.layers.Input(shape=(1,))
input_velocity = tf.keras.layers.Input(shape=(1,))

# Combine inputs using Concatenate layer
concatenated = tf.keras.layers.Concatenate()([input_x, input_velocity])
#normalization layer
normalized = tf.keras.layers.Normalization()(concatenated)
# Dense layers (with multi input it is suggested this configuration: 64->32->16)
dense1 = tf.keras.layers.Dense(64, activation='relu')(normalized)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)

# Output layer
output = tf.keras.layers.Dense(1)(dense3)

# Create model using defined inputs and output
model = tf.keras.models.Model(inputs=[input_x, input_velocity], outputs=output)
model.summary()

print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',  # Use mean squared error for regression
              metrics=['mse'])  # Use mse for evaluation
print("Model compiled successfully")

# Allena e valuta il modello 
EPOCHS = 100
BATCH_SIZE = 32 

print("Fitting model...")
history = model.fit((x_train, velocity_train), V_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=((x_val, velocity_val), V_val))

print("Making predictions...")
predictions = model.predict([x_test, velocity_test])
print("Predictions:", predictions)

prediction_tot_dataset = model.predict([x_data, velocity_data])
plt.plot(prediction_tot_dataset)
plt.xlabel('Prediction sample')  
plt.ylabel('Predicted value')  
plt.title('Predictions on the whole dataset')     
plt.grid(True)   
plt.show()  

#salva su file i valori predetti 
np.savez('PredictionsForActor.npz', prediction_tot_dataset)