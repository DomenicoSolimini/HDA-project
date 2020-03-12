import numpy as np
import argparse
import pickle
import os
import re
import matplotlib.pyplot as plt

# Ignores future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_PATH = "data/gt_tracks/"
OUT_CLEAN_PATH = "data/gt_tracks_clean/"


def import_data(data_path=DATA_PATH, shape=None, axis=0):

    # Init dictionary
    data = {}
    # Save in a list all files present in the folder
    DirList = os.listdir(data_path)
    # Loop over all files contained in the fold
    for track_path in DirList:
        # Extract track name and type (radar or ground truth)
        track_type, track_name = track_path.split('.')[0].split('_', 1)
        # Extract track values
        track = np.load(data_path + track_path)
        # Populate dict values for each track_name key
        if track_name in data:
            # If key exists, add info of the second track
            data[track_name][track_type] = track
        else:
            # If key does not exist, define the value as a new dict
            data[track_name] = {track_type: track}

    return data



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
    predicted_data = model.predict(np.expand_dims(splitted_data, axis= 3)).squeeze()
    # Rescale the output
    predicted_data = predicted_data * (maxs-mins) + mins

    distance = np.concatenate((np.flip(predicted_data[0, :, 0]), predicted_data[1:, 0, 0]))

    angle_vel = np.concatenate((predicted_data[0, :, 1:], predicted_data[1:, 14, 1:]))
    return np.concatenate((distance[:,np.newaxis], angle_vel), axis=1)



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



def convert_to_cartesian(track_radar):
    x = track_radar[:,0]*np.cos(track_radar[:,1])
    y = track_radar[:,0]*np.sin(track_radar[:,1])

    return np.array((x,y))



def compute_RMSE(track1, track2):
    return np.sqrt(np.mean(np.linalg.norm(track1 - track2, axis = 0, ord = 2)**2))



def main():
    # Load predictor from file
    with open('autoencoder.pkl', 'rb') as file:
        predictor = pickle.load(file)
    # Define input argument (name of the track to be selected)
    parser = argparse.ArgumentParser()

    parser.add_argument('--track', type=str, default= "circle_3")
    #parser.add_argument('--plot', type=)
    args = parser.parse_args()
    # Load track data
    tracks = import_data()
    # Select one track
    track_radar = tracks[args.track]['radar1trg']
    track_gt = tracks[args.track]['truth1trg']
    # Compute minimum and maximum for each component
    min_ = np.min(track_radar, axis=0)
    max_ = np.max(track_radar, axis=0)
    # Compute the normalized track component-wise
    track_radar_norm = (track_radar - min_) / (max_ - min_)
    # Compute the predicted track and renormalize it
    track_radar_pred = predict_track(track_radar_norm, predictor) * (max_ - min_) + min_
    # Plot the comparison between radar tracks (measured and predicted)
    # plot_comparison(track_radar, track_radar_pred, var_index=0)
    # Convert the track in cartesian coordinates and compute the RMSD
    rmse = compute_RMSE(convert_to_cartesian(track_radar_pred), track_gt.T)
    # Print RMSE metric
    print(args.track, '\t RMSE: %.3f' % rmse)
    # Save clean track
    np.save(OUT_CLEAN_PATH + args.track + ".npy", track_radar_pred)



if __name__ == "__main__":
    main()
