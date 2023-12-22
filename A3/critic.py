import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
import time
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx):
    ''' Create the neural network to represent the Q function '''
    weight_decay = 0.0009676325797855587
    # OPTION 1
    
    inputs = layers.Input(shape=(nx,1))
    # Add normalization layer
    normalized_inputs = layers.LayerNormalization()(inputs)
    state_out1 = layers.Dense(16, activation="relu")(normalized_inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 
    
    '''
    #OPTION 2
    inputs = layers.Input(shape=(nx,1))
    normalized_inputs = layers.LayerNormalization()(inputs)
    state_out1 = layers.Dense(16, activation="relu")(normalized_inputs)
    state_out2 = layers.Dense(8, activation="relu")(state_out1)    
    outputs = layers.Dense(1)(state_out2)
    '''
    
    '''
    #OPTION 3   -> migliore finora con batch di 15
    inputs = layers.Input(shape=(nx,1))
    normalized_inputs = layers.LayerNormalization()(inputs)
    state_out1 = layers.Dense(16, activation="relu")(normalized_inputs)
    state_out2 = layers.Dense(8, activation="relu")(state_out1) 
    state_out3 = layers.Dense(8, activation="relu")(state_out2)   
    outputs = layers.Dense(1)(state_out3)
    '''
    #OPTION 4 -> suggerimento di optuna
    
    '''
    inputs = layers.Input(shape=(nx,1))
    normalized_inputs = layers.LayerNormalization()(inputs)
    state_out1 = layers.Dense(18, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(normalized_inputs) 
    state_out2 = layers.Dense(43, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(state_out1) 
    state_out3 = layers.Dense(7, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(state_out2) 
    outputs = layers.Dense(1)(state_out3) 
    '''
    
    model = tf.keras.Model(inputs, outputs)

    return model

def update(x_batch, target_values):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Compute batch of Values associated to the sampled batch of states
        V_value = V(x_batch, training=True)                         
        # loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        V_loss = tf.math.reduce_mean(tf.math.square(target_values - V_value)) 
        #V_loss = tf.keras.losses.MeanSquaredError()

    # Compute the gradients of the loss w.r.t. network's parameters (weights and biases)
    V_grad = tape.gradient(V_loss, V.trainable_variables)          
    # Update the critic backpropagating the gradients
    optimizer.apply_gradients(zip(V_grad, V.trainable_variables))
    train_acc_metric.update_state(target_values, V_value)  
     
    return V_loss

def val_step(x_batch, V_batch):
    val_logits = V(x_batch, training=False)
    val_acc_metric.update_state(V_batch, val_logits)

nx = 2
nu = 1
VALUE_LEARNING_RATE = 1e-3
#VALUE_LEARNING_RATE = 2.4e-5  #suggested by optuna

# Create critic NNs
V = get_critic(nx)
V.summary()

# Set optimizer specifying the learning rates
optimizer = tf.keras.optimizers.Adam(VALUE_LEARNING_RATE)

# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn = tf.math.reduce_mean(tf.math.square(target_values - V_value)) 

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

#prepare the training dataset
num_epochs = 26
batch_size = 128

#x_train = data['x_init']
#V = data['V']
#path = tf.keras.utils.get_file('results.npz')
with np.load('results.npz') as data:
    x_train = data['x_init']
    V_train = data['V']

NUM_TRAIN = 12000 # 60% of states for training
NUM_VAL_TEST = 4000 #20% of states for validation and 20% for testing
#reserve 200 samples for validation
x_val = x_train[-NUM_VAL_TEST*2:-NUM_VAL_TEST]
V_val = V_train[-NUM_VAL_TEST*2:-NUM_VAL_TEST]

#reserve 200 samples for test 
x_test = x_train[-NUM_VAL_TEST:]
V_test = V_train[-NUM_VAL_TEST:]
#training set
x_train = x_train[:-NUM_VAL_TEST*2]
V_train = V_train[:-NUM_VAL_TEST*2]

#prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, V_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

#prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, V_val))
val_dataset = val_dataset.batch(batch_size)

#prepare the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, V_test))
test_dataset = test_dataset.batch(batch_size)

print("x_init shape:", x_train.shape)
print("V shape:", V_train.shape)

print("Starting training loop")
tot = []
# Training loop
for epoch in range(num_epochs):
    epoch_losses = []  # To store losses for each batch within the epoch
    print(f"\nStart of epoch {epoch+1}" )
    start_time = time.time()

    # for each epoch open a for loop on the set of data 
    for i, (x_batch_train, V_batch_train) in enumerate(train_dataset):
        x_batch = np2tf(x_batch_train)  # Convert numpy to tensorflow
        V_batch = np2tf(V_batch_train)
        V_loss = update(x_batch, V_batch)
        epoch_losses.append(V_loss.numpy())  # Store the loss for this batch
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    
    # Run a validation loop at the end of each epoch.
    print(f"Validation of epoch {epoch+1}")
    for x_batch_val, V_batch_val in val_dataset:
        x_batch = np2tf(x_batch_val)  # Convert numpy to tensorflow
        V_batch = np2tf(V_batch_val)
        val_step(x_batch, V_batch)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
    
    # Compute epoch loss as the average of all batch losses in this epoch
    epoch_loss = np.mean(epoch_losses)
    # Print epoch information
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    tot.append(epoch_loss)
    
# Plotting the data of losses during training
plt.plot(tot)
plt.xlabel('# epoch')  
plt.ylabel('Loss')  
plt.title('Loss during training')     
plt.grid(True)   
plt.show()      

   

# Testing loop to test the performance of the network with unseen data
test_losses = []
for x_batch_test, V_batch_test in test_dataset:
    x_batch = np2tf(x_batch_test)  # Convert numpy to tensorflow
    V_batch = np2tf(V_batch_test)
    test_loss = update(x_batch, V_batch)
    test_losses.append(test_loss.numpy())

# Compute test loss as the average of all test batch losses
test_loss = np.mean(test_losses)
print(f"Test Loss: {test_loss}")

w = V.get_weights()
for i in range(len(w)):
    print("Shape V weights layer", i, w[i].shape)
    
for i in range(len(w)):
    print("Norm V weights layer", i, np.linalg.norm(w[i]))
    
print("\nDouble the weights")
for i in range(len(w)):
    w[i] *= 2
V.set_weights(w)

w = V.get_weights()
for i in range(len(w)):
    print("Norm V weights layer", i, np.linalg.norm(w[i]))

print("\nSave NN weights to file (in HDF5)")
V.save_weights("weights.h5")

print("Load NN weights from file\n")
V.load_weights("weights.h5")

w = V.get_weights()
for i in range(len(w)):
    print("Norm V weights layer", i, np.linalg.norm(w[i]))
    
