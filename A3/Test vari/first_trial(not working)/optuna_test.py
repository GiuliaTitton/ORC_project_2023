import optuna
import tensorflow as tf
import numpy as np

def objective(trial):
    NUM_TRAIN = 12000  # 60% of states for training
    # Define the hyperparameters to optimize
    n_layers = trial.suggest_int("n_layers", 1, 3)
    num_units = [trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)]
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 10, 50)

    # Create the model with hyperparameters from the trial
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        model.add(
            tf.keras.layers.Dense(
                num_units[i],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    # Prepare the datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(batch_size)

    # Train the model
    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size, validation_data=val_dataset, verbose=0)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_dataset)

    return val_loss


def prepare_datasets(batch_size):
    with np.load('results.npz') as data:
        x_train = data['x_init']
        V_train = data['V']
    # Define NUM_VAL_TEST
    NUM_VAL_TEST = 4000

    # Reserve 200 samples for validation
    x_val = x_train[-NUM_VAL_TEST * 2:-NUM_VAL_TEST]
    V_val = V_train[-NUM_VAL_TEST * 2:-NUM_VAL_TEST]

    # Reserve 200 samples for test
    x_test = x_train[-NUM_VAL_TEST:]
    V_test = V_train[-NUM_VAL_TEST:]
    # Training set
    x_train = x_train[:-NUM_VAL_TEST * 2]
    V_train = V_train[:-NUM_VAL_TEST * 2]
    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, V_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, V_val))
    val_dataset = val_dataset.batch(batch_size)
    # Prepare the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, V_test))
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
