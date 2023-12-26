# starting from the critic network create the actor network, update both networks during training, and then proceed with using these networks for control or solving an Optimal Control Problem (OCP)

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
import time
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from ocp1 import OcpSingleIntegrator as si

# time step
dt = 0.1
# discretization of joint torque
u_max = 1
n_u = 100
u_vector = np.zeros(n_u+1)
for i in range(n_u+1):
    u_vector[i] = -u_max + i*2*u_max/n_u
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

# --------------------CRITIC-------------------------#

def load_critic(nx):
    # load the critic model from saved weights
    # critic architecture (option 4)
    weight_decay = 0.0009676325797855587
    inputs = layers.Input(shape=(nx,1))
    state_out1 = layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs) 
    state_out2 = layers.Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(state_out1) 
    state_out3 = layers.Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(state_out2) 
    outputs = layers.Dense(1,)(state_out3) 
    model_critic = tf.keras.Model(inputs, outputs)
    # critic weights 
    model_critic.load_weights('weights.h5')
    return model_critic

#-------------------ACTOR----------------------#

def get_actor(nx):
    ''' Create the neural network to represent the actor '''
    # 1 input -> greedy policy w.r.t. critic
    inputs = layers.Input(shape=(nx,1))
    normalized_inputs = layers.LayerNormalization()(inputs)
    state_out1 = layers.Dense(16, activation="relu")(normalized_inputs) 
    state_out2 = layers.Dense(8, activation="relu")(state_out1) 
    state_out3 = layers.Dense(8, activation="relu")(state_out2) 
    outputs = layers.Dense(1)(state_out3)
    
    model = tf.keras.Model(inputs, outputs)

    return model

def update_actor(x_batch, target_values):
    ''' Update the weights of the Actor network using the specified batch of data '''
    # Perform actor network training similarly to how you update the critic network
    # Use a loss function based on the greedy policy w.r.t. critic
    with tf.GradientTape() as tape:
        actions_pred = actor(x_batch, training=True)

        #actor_loss = -tf.reduce_mean(action_value)
        actor_loss = tf.math.reduce_mean(tf.math.square(target_values - actions_pred)) 

    actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))
    
    return actor_loss, action_value

# compute running cost
def l(x,u):
    x_term = (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1)
    u_term = 0.5 * u**2 
    cost = u_term + x_term 
    return cost

def val_step(x_batch, pi):
    val_logits = V(x_batch, training=False)
    val_acc_metric.update_state(pi, val_logits)

nx = 1
nu = 1
#VALUE_LEARNING_RATE = 1e-3
VALUE_LEARNING_RATE = 2.4e-5  #suggested by optuna

# Load the critic
model_critic = load_critic(nx)

# Initialize the actor network
actor = get_actor(nu)
actor.summary()

# Set optimizer specifying the learning rates
optimizer = tf.keras.optimizers.Adam(VALUE_LEARNING_RATE)

# Instantiate a loss function.
loss_fn = tf.keras.losses.MeanSquaredError()

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.MeanSquaredError()
val_acc_metric = tf.keras.metrics.MeanSquaredError()

# Prepare the training dataset
num_epochs = 10
batch_size = 15

with np.load('results.npz') as data:
    x_train = data['x_init']

NUM_TRAIN = 12000 # 60% of states for training
NUM_VAL_TEST = 4000 # 20% of states for validation and 20% for testing
# reserve 200 samples for validation
x_val = x_train[-NUM_VAL_TEST*2:-NUM_VAL_TEST]
#reserve 200 samples for test 
x_test = x_train[-NUM_VAL_TEST:]
#training set
x_train = x_train[:-NUM_VAL_TEST*2]

#prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

#prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
val_dataset = val_dataset.batch(batch_size)

#prepare the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.batch(batch_size)

print("Starting training loop")
tot = []
tot_loss = []
tot_pe = []
total_time_for_training = 0
actor_losses_tot = []
#---------------TRAINING---------------#
# Training loop
for epoch in range(num_epochs):
    epoch_losses = []  # To store losses for each batch within the epoch
    print(f"\nStart of epoch {epoch+1}" )
    start_time = time.time()
    actor_losses = []

    # for each epoch open a for loop on the set of data 
    for i, (x_batch_train) in enumerate(train_dataset):
        x_batch = np2tf(x_batch_train)  # Convert numpy to tensorflow
        pi = np.zeros(len(x_batch))
        for k in range(len(x_batch)):
            V_pred = np.zeros(len(u_vector))
            Q = np.zeros(len(u_vector))
            for j in range(len(u_vector)):
                # predict V of the next state using the critic and save Q for each u
                x_next = x_batch[k] + dt*u_vector[j]
                V_pred[j] = model_critic.predict(x_next)
                Q[j] = l(x_batch[k], u_vector[j]) + V_pred[j]
            # find greedy policy minimizing Q
            pi_pos = np.argmin(Q)
            pi[k] = u_vector[pi_pos]

        actor_loss, action_value = update_actor(x_batch,pi)
        actor_losses.append(actor_loss) 

    actor_losses_tot.append(np.min(actor_losses))
    print(f"Actor loss = {np.min(actor_losses)}")
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    #print("Training acc over epoch: %.4f" % (float(train_acc),))
    
    # Run a validation loop at the end of each epoch.
    print(f"Validation of epoch {epoch+1}")
    for x_batch_val in val_dataset:
        x_batch = np2tf(x_batch_val)  # Convert numpy to tensorflow
        val_logits = val_step(x_batch, pi)
        
        actor_predictions = actor(val_logits, training=False)
        
        val_acc_metric.update_state(pi, actor_predictions)

    val_acc = val_acc_metric.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    # Reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    
    time_taken = (time.time() - start_time)
    print("Time taken: %.2fs" % time_taken)
    total_time_for_training += time_taken
    
    # Compute epoch loss as the average of all batch losses in this epoch
    epoch_loss = np.mean(epoch_losses)
    # Print epoch information
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    tot.append(epoch_loss)

print(f"Total time taken for training: {total_time_for_training}")
    
# Plotting the data of losses during training
plt.plot(tot)
plt.xlabel('# epoch')  
plt.ylabel('Loss')  
plt.title('EPOCH Loss during training')     
plt.grid(True)   
plt.show()      

plt.plot(actor_losses_tot)
plt.xlabel('')  
plt.ylabel('Actor Loss')  
plt.title('Actor loss during training')     
plt.grid(True)   
plt.show() 

#---------------TEST-----------------#
# Testing loop to test the performance of the network with unseen data
test_losses_actor = []
for x_batch_test in test_dataset:
    x_batch = np2tf(x_batch_test)  # Convert numpy to tensorflow
    pi = np.zeros(len(x_batch))
    for k in range(len(x_batch)):
        V_pred = np.zeros(len(u_vector))
        Q = np.zeros(len(u_vector))
        for j in range(len(u_vector)):
            # predict V of the next state using the critic and save Q for each u
            x_next = x_batch[k] + dt*u_vector[j]
            V_pred[j] = model_critic.predict(x_next)
            Q[j] = l(x_batch[k], u_vector[j]) + V_pred[j]
        # find greedy policy minimizing Q
        pi_pos = np.argmin(Q)
        pi[k] = u_vector[pi_pos]

    
    actor_loss, actor_value = update_actor(x_batch, pi)
    test_losses_actor.append(actor_loss)
# Compute test loss as the average of all test batch losses

test_losses_actor = np.mean(test_losses_actor)
print(f"Test Loss Actor: {test_losses_actor}")

w = actor.get_weights()
for i in range(len(w)):
    print("Shape pi weights layer", i, w[i].shape)
    
for i in range(len(w)):
    print("Norm pi weights layer", i, np.linalg.norm(w[i]))

print("\nSave NN weights to file (in HDF5)")
actor.save_weights("weights_actor.h5")
