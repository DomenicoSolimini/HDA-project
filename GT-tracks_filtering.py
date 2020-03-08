import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle


DATA_PATH = "gt_tracks/"



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
    for j, track_path in enumerate(DirList):

        track = np.load(data_path + track_path)

        min_ = np.min(track, axis=0)
        max_ = np.max(track, axis=0)

        norm_track = (track - min_) / (max_ - min_)

        track_data = {'format': track_path.split('_')[0],
                      'shape_': track_path.split('_')[1],
                      'number': int(track_path.split('_')[2][0]),
                      'track': track,
                      'track_norm': norm_track,
                      'min': min_,
                      'max': max_}

        #track = min_max_scale(track)
        data.append(track_data)

        # print('Track {} saved'.format(track_path))

    return data #pd.DataFrame(data)



def split(data, width=15, stride=5):
    """
    Divide the sequences stored in input array in frame of assigned width shifted of 'stride'.
    Return the frames stacked in a new array. If stride equals width, no overlap is created.
    """
    # Get tracks lenght
    track_len = data.shape[0]

    data_ = [data[start:start+width, :] for start in range(0, track_len-width+1, stride)]
    return np.stack(data_, axis=0)



def predict_track(data, model, frame_width=15):
    """
    Select one track and compute the denoised version using the specified model
    The scaling part still needs to be vectorized.
    """
    # Split data adding a new point for each new window
    splitted_data = split(data, width=frame_width, stride=1)
    # Init the vector to store values for rescaling the input after the training
    # Scale the windows (same factor for every component)
    mins = splitted_data.min(axis=(1,2), keepdims=True)
    maxs = splitted_data.max(axis=(1,2), keepdims=True)
    splitted_data = (splitted_data-mins)/(maxs-mins)
    # Compute the predicted data for each splitted frame
    predicted_data = model.predict(splitted_data)
    # Rescale the output
    predicted_data = predicted_data * (maxs-mins) + mins

    return np.concatenate((predicted_data[0], predicted_data[1:, 14, :]))



def plot_comparison(track_original, track_predicted, var_index, savefig_path=False):
    """
    Select one track, one component of that track and plot the original track and the denoised one.
    """
    # Plot the predicted serie alongside the original one
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(18,6))
    for var_index in range(3):
        ax[var_index].scatter(range(track_predicted.shape[0]), track_predicted[:, var_index], marker='.')
        ax[var_index].scatter(range(track_original.shape[0]), track_original[:, var_index], marker='+')
        ax[var_index].legend(('predicted', 'true'))
        ax[var_index].grid()

    if savefig_path:
        plt.savefig(savefig_path)
    plt.show()

    return None



def convert_to_cartesian(track):
    x = track[:,0]*np.cos(track[:,1])
    y = track[:,0]*np.sin(track[:,1])

    return np.array((x,y))



if __name__ == "__main__":

    with open('autoencoder.pkl', 'rb') as file:
        predictor = pickle.load(file)

    tracks_list = import_data()
    track = tracks_list[3]

    print(track['format'], track['shape_'], track['number'], '\n\n')

    track_predicted_norm = predict_track(track['track_norm'], predictor)
    track_predicted = track_predicted_norm * (track['max'] - track['min']) + track['min']

    plot_comparison(track['track'], track_predicted, var_index=0)
