import numpy as np
import casadi
import ocp1
import tensorflow as tf
import tensorflow_example as tf_template
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform


# NEURAL NETORK TO PREDICT THE COST V GIVEN AN INITAL STATE X0

# Loading data from a Numpy .npz file
#data = np.load('results.npz')

#prepare the training dataset
batch_size = 25 # number of data samples (examples) processed in one iteration of the neural network  
# Choosing an appropriate batch size depends on several factors, including the size of your dataset, available memory, computational resources, and the specific characteristics of your problem. It's a hyperparameter that you might need to tune while training your neural network.

#x_train = data['x_init']
#V = data['V']
#path = tf.keras.utils.get_file('results.npz')
with np.load('results.npz') as data:
    x_train = data['x_init']
    V_train = data['V']

#reserve 200 samples for validation
x_val = x_train[-400:-200]
V_val = V_train[-400:-200]
#reserve 200 samples for test 
x_test = x_train[-200:]
V_test = V_train[-200:]
#training set
x_train = x_train[:-400]
V_train = V_train[:-400]

#prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, V_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

#prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, V_val))
val_dataset = val_dataset.batch(batch_size)

#prepare the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, V_test))
test_dataset = test_dataset.batch(batch_size)

# Check the shape and content of the loaded arrays
print("x_init shape:", x_train.shape)
print("V shape:", V_train.shape)
'''
# Print a few samples to inspect the data
print("x_init samples:", x_train[:10])
print("V samples:", V[:10])

# Check if the lengths of x_init and V match
if len(x_train) == len(V):
    print("Data lengths are aligned.")
else:
    print("Data lengths are not aligned. Investigate further.")
'''

# Training parameters

num_epochs = 100  # number of times the entire dataset is passed forward and backward through the neural network during the training process
#using too few epochs may result in the model not learning enough from the data, leading to underfitting (poor performance on both training and unseen data). Conversely, using too many epochs might lead to overfitting, where the model performs well on the training data but fails to generalize to new, unseen data. The number of epochs is one of the hyperparameters that you may need to tune during the training process to find the right balance between underfitting and overfitting.

# Training loop
for epoch in range(num_epochs):
    epoch_losses = []  # To store losses for each batch within the epoch

    for i in range(0, len(x_train), batch_size):
        x_batch = tf_template.np2tf(x_train[i:i+batch_size])  # Convert numpy to tensorflow
        target_values = tf_template.np2tf(V_train[i:i+batch_size])  # Convert numpy to tensorflow

        # Perform update on the batch
        batch_loss = tf_template.update(x_batch, target_values)
        epoch_losses.append(batch_loss.numpy())  # Store batch loss

    # Compute epoch loss as the average of all batch losses in this epoch
    epoch_loss = np.mean(epoch_losses)
    # Print epoch information
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")


# Save the trained model
V_train.save_weights("trained_model.h5")


# NB: Convergence and Stability: Focus on achieving convergence and stability in the epoch loss. The loss might stabilize at a certain level, indicating your model's understanding of the dynamics without necessarily reaching near-zero values.striving for a lower epoch loss is essential, aiming for near-zero loss might be unrealistic due to the system's complexity and chaotic nature