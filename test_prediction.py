import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from build_predictor import min_max_scale

DATA_PATH = './gt_tracks/'
DATA_PATH = './all_tracks/'


def import_data(data_path=DATA_PATH, shape=None, axis=0):
    """
    Import data stored in a folder identified by its path.
    Return a list containing all the arrays present in the folder.
    """
    # Init container
    data = []
    # Save in a list all files present in the folder
    DirList = os.listdir(data_path)
    #
    for j, file in enumerate(DirList):
        track = np.load(data_path + file)
        track = min_max_scale(track)
        data.append(track)

    return data



def predict_track(data, model, track_index, frame_width):
    """
    Select one track and compute the denoised version using the specified model
    The scaling part still needs to be vectorized.
    """
    # Select only one track and add a supplementary dimension on the first axis
    data = np.expand_dims(data[track_index], axis=0)
    # Split data adding a new point for each new window
    splitted_data = split(data, width=frame_width, stride=1)
    # Init the vector to store values for rescaling the input after the training
    min_max_vec = []
    # Scale the windows (same factor for every component)
    for frame in splitted_data:
        min_max_vec.append({'min': np.min(frame), 'max': np.max(frame)})
        frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame))
    # Compute the predicted data for each splitted frame
    predicted_data = model.predict(np.expand_dims(splitted_data, axis=3)).squeeze()
    # Rescale the output
    for i, prediction in enumerate(predicted_data):
        prediction = prediction*(min_max_vec[i]['max'] - min_max_vec[i]['min']) + min_max_vec[i]['min']

    # Pad undefined values with nans
    predicted_data = np.pad(predicted_data, ((13,0),(0,0)), 'constant', constant_values=np.nan)

    return predicted_data



def plot_comparison(x_test, predicted, track_index, var_index, savefig=False):
    """
    Select one track, one component of that track and plot the original track and the denoised one.
    """
    # Plot the predicted serie alongside the original one
    fig = plt.figure(figsize=(18,6))
    plt.scatter(range(predicted.shape[1]), predicted[var_index], marker='.')
    plt.scatter(range(x_test.shape[1]), x_test[track_index,:,var_index], marker='+')
    # plt.xlim(70, 110)
    plt.legend(('predicted', 'true'))
    plt.grid()

    if savefig:
        plt.savefig('plots/dim{}.png'.format(track_index))
    plt.show()

    return None


import_data()

def main(track_index, frame_width):
    prediction = predict_track(data, model, track_index, frame_width)
    plot_comparison(x_test, predicted, track_index, var_index)
