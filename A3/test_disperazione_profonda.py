import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Carica set di dati
data = np.load('results.npz')
x_data = data['x_init']
V_data = data['V']

# Crea dataset e mescola i dati
dataset = tf.data.Dataset.from_tensor_slices((x_data, V_data))
dataset_shuffle = dataset.shuffle(buffer_size=20000)
#for element in dataset_shuffle.as_numpy_iterator():
#  print(element)

# Crea train, test e validation dataset 
train_dataset = dataset_shuffle.take(14000)

start_index_test = 17000
test_dataset = dataset_shuffle.skip(start_index_test).take(3000)

start_index_val = 14000
val_dataset = dataset_shuffle.skip(start_index_val).take(3000)

# Estrai x e V
x_train = []
V_train = []

for x, V in train_dataset:
    x_train.append(x.numpy())
    V_train.append(V.numpy())

x_train = np.array(x_train)
V_train = np.array(V_train)

nx = 1
# Costruisci modello di apprendimento automatico
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 1)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
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
EPOCHS = 20
BATCH_SIZE = 32

print("Fitting model...")
history = model.fit(x_train, V_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_dataset)

print("Making predictions...")
predictions = model.predict(x_test)
print("Predictions:", predictions)
