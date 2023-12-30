import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carica x_data
data = np.load('resultsDoubleInt.npz')
x_data = data['x_init']
vel_data = data['v_init']

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

pi = np.zeros(len(x_data))
print("Computing pi ...")
# For each initial state
for i in range(len(x_data)):
    x_next = np.zeros(len(u_vector))
    v_next =np.zeros(len(u_vector))
    # Find possible x_next, v_next depending on u
    for j in range(len(u_vector)):
        x_next[j] = x_data[i] + dt*vel_data[i] + 0.5* (dt**2) *u_vector[j]
        v_next[j] = vel_data[i] + dt*u_vector[j]
        
    # Predict V(x_next) for each possible u
    V_pred_dataset = model_critic.predict((x_next,v_next), verbose=0)
    # Transform V_pred from dataset to numpy array
    V_pred = np.array(V_pred_dataset)
    V_pred = V_pred.flatten()
    # Find greedy policy minimizing Q
    Q = np.zeros(len(u_vector))
    for j in range(len(u_vector)):
        Q[j] = l(x_data[i], u_vector[j]) + V_pred[j]
    pi_pos = np.argmin(Q)
    pi[i] = u_vector[pi_pos]

plt.plot(pi)
plt.xlabel('Prediction number (initial states)')  
plt.ylabel('Optimal control')  
plt.title('Optimal control based on critic predictions')     
plt.grid(True)   
plt.show()  


# Crea dataset e mescola i dati
dataset = tf.data.Dataset.from_tensor_slices((x_data,vel_data, pi))
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
v_train = []
pi_train = []

for x, v, pi in train_dataset:
    x_train.append(x.numpy())
    v_train.append(v.numpy())
    pi_train.append(pi.numpy())

x_train = np.array(x_train)
v_train = np.array(v_train)
pi_train = np.array(pi_train)

# Estrai x_val e V_val come numpy array
x_val = []
v_val = []
pi_val = []

for x, v, pi in val_dataset:
    x_val.append(x.numpy())
    v_val.append(v.numpy())
    pi_val.append(pi.numpy())

x_val = np.array(x_val)
v_val = np.array(v_val)
pi_val = np.array(pi_val)

# Estrai x_test e V_test come numpy array
x_test = []
v_test = []
pi_test = []

for x, v, pi in test_dataset:
    x_test.append(x.numpy())
    v_test.append(v.numpy())
    pi_test.append(pi.numpy())

x_test = np.array(x_test)
v_test = np.array(v_test)
pi_test = np.array(pi_test)

nx = 1
'''
# Costruisci modello di apprendimento automatico
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])
model.summary()'''

print("Creating model ...")
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
print("Model actor:")
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
history = model.fit((x_train,v_train), pi_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, pi_val))

print("Making predictions...")
predictions = model.predict([x_test,v_test])
print("Predictions:", predictions)

prediction_tot_dataset = model.predict([x_data,vel_data])
plt.plot(prediction_tot_dataset)
plt.xlabel('Prediction number (initial states)')  
plt.ylabel('Predicted value (optimal control)')  
plt.title('Predictions on the whole dataset')     
plt.grid(True)   
plt.show()  

#salva su file i valori predetti 
np.savez('PredictionsForOCPDoubleIntegrator.npz', prediction_tot_dataset)