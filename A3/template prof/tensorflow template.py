import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform

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
    inputs = layers.Input(shape=(nx,1))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 

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
    # Compute the gradients of the loss w.r.t. network's parameters (weights and biases)
    V_grad = tape.gradient(V_loss, V.trainable_variables)          
    # Update the critic backpropagating the gradients
    optimizer.apply_gradients(zip(V_grad, V.trainable_variables))    

nx = 2
nu = 1
VALUE_LEARNING_RATE = 1e-3

# Create critic NNs
V = get_critic(nx)
V.summary()

# Set optimizer specifying the learning rates
optimizer = tf.keras.optimizers.Adam(VALUE_LEARNING_RATE)

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
V.save_weights("namefile.h5")

print("Load NN weights from file\n")
V.load_weights("namefile.h5")

w = V.get_weights()
for i in range(len(w)):
    print("Norm V weights layer", i, np.linalg.norm(w[i]))
    