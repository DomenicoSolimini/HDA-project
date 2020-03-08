import os
import numpy as np
import pickle

DATA_PATH = './all_tracks/'
OUT_NOISY_PATH = 'data_noisy.npy'
OUT_AUG_PATH = 'data_augmented.npy'


def import_data(data_path=DATA_PATH, shape=None, axis=0):
    """
    Import data stored in a folder identified by its path.
    Return a numpy array containing all the array present in the folder of specified shape
    stacked along the specified axis. If shape is None, the shape of the first element is used.
    """
    # Init container
    MyData = []
    # Save in a list all files present in the folder
    DirList = os.listdir(data_path)

    # Set the shape variable
    if shape == None:
        shape = np.load(data_path + DirList[0]).shape

    #
    for j in range(len(DirList)):
        data = np.load(data_path + DirList[j])
        if data.shape == shape:
            MyData.append(data)
        else:
            print('MyData[{}] has invalid shape {}'.format(j, data.shape))

    # Stack togheter all the arrays
    data = np.stack(MyData, axis = axis)

    return data



def min_max_scale(data):
    """
    Scale separately each component. Data has shape [num_frames, frame_width, num_components].
    """
    # Compute separately min and max for each component
    mins = data.min(axis=1, keepdims=True)
    maxs = data.max(axis=1, keepdims=True)
    # Return normalized data
    return (data - mins) / (maxs - mins)



def GenerateNoisyData(data, seed, variance):
    """ Return Data with a gaussian noise of mean=0 and var=0.05
    """
    np.random.seed(seed)
    noise = np.random.normal(loc = 0, scale = np.sqrt(variance), size = data.shape)
    return data + noise



def main(variance):
    data = import_data()
    data = min_max_scale(data)

    noisy_data_1 = GenerateNoisyData(data, 1, variance)
    noisy_data_2 = GenerateNoisyData(data, 2, variance)

    data_noisy = np.concatenate((data, noisy_data_1, noisy_data_2), axis=0)
    data_augmented = np.concatenate((data, data, data), axis=0)

    np.save(OUT_NOISY_PATH, data)
    print('Data noisy saved in ', OUT_NOISY_PATH)

    np.save(OUT_AUG_PATH, data)
    print('Data augmented saved in ', OUT_AUG_PATH)



if __name__ == "__main__":
    main(0.05)
