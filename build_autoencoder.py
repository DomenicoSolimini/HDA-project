import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Dropout

DATA_AUG_PATH = 'data_augmented.pkl'
DATA_NOISY_PATH = 'data_noisy.pkl'
MODEL_PATH = 'autoencoder.pkl'
TRACK_SHAPE = (496, 3)


def split(data, width=15, stride=1):
    """
    Divide the sequences stored in input array in frame of assigned width shifted of 'stride'.
    Return the frames stacked in a new array. If stride equals width, no overlap is created.
    """
    # Get tracks lenght
    track_len = data.shape[1]

    data_ = [data[:, start:start+width, :] for start in range(0, track_len-width+1, stride)]
    return np.concatenate(data_, axis=0)



def load_data(data_aug_path, data_noisy_path, shape):
    """
    """
    # Import data
    with open(data_noisy_path, 'rb') as infile:
        x_data = pickle.load(infile)

    with open(data_aug_path, 'rb') as infile:
        y_data = pickle.load(infile)
    # Divide in Train, Validation and Test sets and separately scale data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, random_state=42, test_size=0.4)
    #x_test, x_val, y_tra = train_test_split(x_val, random_state=42, test_size=0.5)  # LEVARE IL TEST SET UNA VOLTA FINITO!!!

    print(x_val.shape, x_train.shape)
    # Split sets in subsequent windows
    x_train, x_val, y_train, y_val = [split(set_, width=15, stride=1) for set_ in (x_train, x_val, y_train, y_val)]

    return x_train, x_val, y_train, y_val



def create_autoencoder():
    input_seq = Input(shape=(15, 3))
    X = Flatten()(input_seq)
    X = Dense(20, activation='tanh')(X)
    encoded = Dense(10, activation='tanh')(X)
    X = Dense(20, activation='tanh')(encoded)
    X = Dense(45, activation='tanh')(X)
    decoded = Reshape((15, 3))(X)

    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer="Nadam", loss="mean_absolute_error")
    autoencoder.summary()

    return autoencoder



def train_model(x_train, x_val, y_train, y_val, model, max_epochs=250, plot=True):
    """
    """
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(x_train, y_train,
                    epochs = max_epochs,
                    steps_per_epoch = None,
                    shuffle = True,
                    batch_size = None,
                    validation_data=(x_val, y_val),
                    callbacks = [es])

    if plot:
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(111)

        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train Set', 'Validation Set'], loc='upper right')
        plt.tight_layout()
        plt.show()

    return model



def main(verbose=False):
    # Load dataset
    x_train, x_val, y_train, y_val = load_data(data_aug_path=DATA_AUG_PATH,
                                               data_noisy_path=DATA_NOISY_PATH,
                                               shape=TRACK_SHAPE)

    # Scale each frame
    for set_ in (x_train, x_val, y_train, y_val):
        mins = set_.min(axis=(1,2), keepdims=True)
        maxs = set_.max(axis=(1,2), keepdims=True)
        set_ = (set_-mins)/(maxs-mins)

    # Print information
    if verbose:
        print('Train set shape \t', x_train.shape)
        print('Validation set shape \t', x_val.shape)
        print('Number of test tracks \t', len(x_test))
    # Create or refresh the neural network
    predictor = create_autoencoder()
    # Train the net
    history = train_model(x_train, x_val, y_train, y_val, model=predictor, max_epochs=250)
    # Save the model
    with open(MODEL_PATH, 'wb') as infile:
        pickle.dump(predictor, infile)

    return predictor



if __name__ == '__main__':
    main()
