import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Dropout

DATA_PATH =  'data_augmented.pkl'
MODEL_PATH = 'autoencoder.pkl'
TRACK_SHAPE = (496, 3)






def train_val_test_split(data, train=0.7, val=0.2):
    """
    Takes in input an array formatted as (num sequences x dim1 x dim2).
    """
    # Get the total sequences number
    n_sequences = data.shape[0]
    # Random shuffle data
    
    # Set indexes to divide the sets
    train_ind = int(n_sequences*train)
    val_ind = train_ind + int(n_sequences*val)

    return data[:train_ind], data[train_ind:val_ind], data[val_ind:]



def split(data, width=15, stride=5):
    """
    Divide the sequences stored in input array in frame of assigned width shifted of 'stride'.
    Return the frames stacked in a new array. If stride equals width, no overlap is created.
    """
    # Get tracks lenght
    track_len = data.shape[1]

    data_ = [data[:, start:start+width, :] for start in range(0, track_len-width+1, stride)]
    return np.concatenate(data_, axis=0)



def load_data(data_path, shape):
    """
    """
    # Import data
    with open(DATA_PATH, 'rb') as infile:
        data = pickle.load(infile)
    # Divide in Train, Validation and Test sets and separately scale data
    x_train, x_val = train_test_split()
    # Split train set in subsequent windows
    x_train = split(x_train, width=15, stride=1)
    # Split validation set in subsequent windows
    x_val = split(x_val, width=15, stride=1)

    return x_train, x_val, x_test
