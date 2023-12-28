import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#carica set di dati
data = np.load('results.npz')
x_data = data['x_init']
V_data = data['V']

x_train = x_data[:16000]
V_train = V_data[:16000]
x_test = x_data[16000:]
V_test = V_data[16000:]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, V_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, V_test))

BATCH_SIZE = 15
EPOCHS=50
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#costruisci modello di apprendimento automatico
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

print("Fitting model...")
#allena e valuta il modello 
history = model.fit(x_train,V_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("Making predictions...")
predictions = model.predict(x_test)
print("Predictions:", predictions)

print("Evaluating model...")
model.evaluate(x_test)

