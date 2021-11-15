from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plotting(history):
    """Plots to visualise model performance. Plotting loss and val_loss as function of epochs"""
    plt.plot(history.history['loss'], color="red")
    plt.plot(history.history['val_loss'], color="blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()


def stateful_cut(arr, batch_size, t_after_cut):
    """Helper function for stateful model"""
    if len(arr.shape) != 3:
        print("ERROR: please format arr as a (N, T, 3) array.")

    N = arr.shape[0]
    T = arr.shape[1]
    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / t_after_cut)
    if nb_cuts * t_after_cut != T:
        print("ERROR: T_after_cut must divide T")

    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return cut4


def define_reset_states_class(nb_cuts):
    """Function to define 'Callback resetting model states' class"""
    class ResetStatesCallback(Callback):
        def __init__(self):
            self.counter = 0

        def on_batch_begin(self, batch, logs={}):
            # We reset states when nb_cuts batches are completed, as
            # shown in the after cut figure
            if self.counter % nb_cuts == 0:
                self.model.reset_states()
                self.counter += 1

        def on_epoch_end(self, epoch, logs={}):
            # reset states after each epoch
            self.model.reset_states()

    return ResetStatesCallback


def batched(i, arr, batch_size):
    """Helper function to get data in batches"""
    return (arr[i * batch_size:(i + 1) * batch_size])


def test_on_batch_stateful(model, inputs, outputs, batch_size, nb_cuts):
    """Helper function to test the model is stateful"""
    nb_batches = int(len(inputs) / batch_size)
    sum_pred = 0
    for i in range(nb_batches):
        if i % nb_cuts == 0:
            model.reset_states()
            x = batched(i, inputs, batch_size)
            y = batched(i, outputs, batch_size)
            sum_pred += model.test_on_batch(x, y)
            mean_pred = sum_pred / nb_batches
    return mean_pred


def define_stateful_val_loss_class(inputs, outputs, batch_size, nb_cuts):
    """
    Function to define 'Callback computing validation loss' class #
    Callback to reset states are not properly called with validation data, as
    noted by Philippe Remy in http://philipperemy.github.io/keras-stateful-lstm :
    "be careful as it seems that the callbacks are not properly called when using
    the parameter validation_data in model.fit(). You may have to do your
    validation/testing manually by calling predict_on_batch() or
    test_on_batch()."

    We introduce a callback to compute validation loss to circumvent this.
    Result will looks like this:
    Epoch 56/100 750/750 [======] - 1s - loss: 1.5133e-04     val_loss: 2.865e-04
    """
    class ValidationCallback(Callback):
        def __init__(self):
            self.val_loss = []

        def on_epoch_end(self, epoch, logs={}):
            mean_pred = test_on_batch_stateful(self.model, inputs, outputs,
                                               batch_size, nb_cuts)
            print('val_loss: {:0.3e}'.format(mean_pred), end='')
            self.val_loss += [mean_pred]

        def get_val_loss(self):
            return (self.val_loss)

    return ValidationCallback



def get_data(data_path):
    """Getting processed data"""
    df = pd.read_csv(data_path, index_col=[0])
    df = df.dropna(subset=['Lat_7_days', 'Lon_7_days', 'CPUE_7_days']).sort_values('Date')
    return df


def data_initial_input_output(df, N, T):
    """Preparing data for model input"""
    round_end = 3840

    # Inputs
    x1_f = df['Lat_7_days'].values[:round_end].reshape(2 * N, T)
    x2_f = df['chl'].values[:round_end].reshape(2 * N, T)
    x3_f = df['Lon_7_days'].values[:round_end].reshape(2 * N, T)
    x4_f = df['thetao'].values[:round_end].reshape(2 * N, T)

    # Outputs
    y1_f = df['CPUE_number_per_hour'].values[:round_end].reshape(2 * N, T)
    y2_f = df['Lat'].values[:round_end].reshape(2 * N, T)
    y3_f = df['Lon'].values[:round_end].reshape(2 * N, T)

    return x1_f, x2_f, x3_f, x4_f, y1_f, y2_f, y3_f


def train_test_data(df, N, T):
    """Splitting data to test/train datasets"""
    x1_f, x2_f, x3_f, x4_f, y1_f, y2_f, y3_f = data_initial_input_output(df, N, T)

    # Training/test sets
    x1_train_f, x2_train_f, x3_train_f, x4_train_f = [x[0:N] for x in [x1_f, x2_f, x3_f, x4_f]]
    x1_test_f, x2_test_f, x3_test_f, x4_test_f = [x[N:2 * N] for x in [x1_f, x2_f, x3_f, x4_f]]

    y1_train_f, y2_train_f, y3_train_f = [y[0:N] for y in [y1_f, y2_f, y3_f]]
    y1_test_f, y2_test_f, y3_test_f = [y[N:2 * N] for y in [y1_f, y2_f, y3_f]]

    # Reshape each time series as (N, T, dim_in) or (N, T, dim_out)
    x_train_f = np.concatenate((x1_train_f.reshape(N, T, 1),
                                x2_train_f.reshape(N, T, 1),
                                x3_train_f.reshape(N, T, 1),
                                x4_train_f.reshape(N, T, 1),
                                ), axis=2)

    y_train_f = np.concatenate((y1_train_f.reshape(N, T, 1),
                                y2_train_f.reshape(N, T, 1),
                                y3_train_f.reshape(N, T, 1),
                                ), axis=2)

    x_test_f = np.concatenate((x1_test_f.reshape(N, T, 1),
                               x2_test_f.reshape(N, T, 1),
                               x3_test_f.reshape(N, T, 1),
                               x4_test_f.reshape(N, T, 1),
                               ), axis=2)

    y_test_f = np.concatenate((y1_test_f.reshape(N, T, 1),
                               y2_test_f.reshape(N, T, 1),
                               y3_test_f.reshape(N, T, 1),
                               ), axis=2)

    return x_train_f, x_test_f, y_train_f, y_test_f


def model_input_otputs(df, N, T, batch_size, t_after_cut):
    """Preparing data in batches for model input"""
    x_train_f, x_test_f, y_train_f, y_test_f = train_test_data(df, N, T)
    inputs, outputs, inputs_test, outputs_test = [stateful_cut(arr, batch_size, t_after_cut) for arr in \
                                                  [x_train_f, y_train_f, x_test_f, y_test_f]]

    return inputs, outputs, inputs_test, outputs_test


def lstm_model(batch_size, dim_in, dim_out):
    """Setting up the model"""
    model = Sequential()
    model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),
                   return_sequences=True, units=100, stateful=True))
    model.add(LSTM(30, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(model, epochs, batch_size, N, T, t_after_cut, inputs, outputs, inputs_test, outputs_test):
    """Training the stateful model"""
    # Model Training
    nb_reset = int(N / batch_size)
    nb_cuts = int(T / t_after_cut)
    if nb_reset > 1:
        ResetStatesCallback = define_reset_states_class(nb_cuts)
        ValidationCallback = define_stateful_val_loss_class(inputs_test, outputs_test, batch_size, nb_cuts)
        validation = ValidationCallback()
        history = model.fit(inputs, outputs, epochs=epochs,
                            batch_size=batch_size, shuffle=False,
                            callbacks=[ResetStatesCallback(), validation])
        history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
    else:
        # If nb_reset = 1, we should reset states after each epoch.
        # To improve computational speed, we can decide not to reinitialize states
        # at all. Results are similar in this case.
        # In the following line, states are not reinitialized at all:
        history = model.fit(inputs, outputs, epochs=epochs,
                            batch_size=batch_size, shuffle=False,
                            validation_data=(inputs_test, outputs_test))
    return model, history


def predict(model, dim_in, dim_out, x_test_f, n):
    """Predicting targets with the trained model"""
    # n = n # time series selected (between 0 and N-1)
    model_stateless = Sequential()
    model_stateless.add(LSTM(input_shape=(None, dim_in), return_sequences=True, units=100))
    model_stateless.add(LSTM(30, activation='relu', return_sequences=True))
    model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model_stateless.compile(loss = 'mse', optimizer = 'rmsprop')
    model_stateless.set_weights(model.get_weights())

    ## Prediction of a new set
    x = x_test_f[n]
    y_hat = model_stateless.predict(np.array([x]))[0]
    return y_hat
