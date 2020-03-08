import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, Flatten, Reshape

DATA_PATH = './all_tracks/'
MODEL_PATH = 'model.pkl'
SHAPE = (496, 3)


def import_data(data_path, shape=None, axis=0):
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



def train_val_test_split(data, train=0.7, val=0.2):
    """
    Takes in input an array formatted as (num sequences x dim1 x dim2).
    """
    # Get the total sequences number
    n_sequences = data.shape[0]
    
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



def min_max_scale(data):
    """
    Scale separately each component. Data has shape [num_frames, frame_width, num_components].
    """
    # Keep the number of time instants
    frame_width = data.shape[1]
    
    # Compute separately min and max for each component
    min_, max_ = data.min(axis=1), data.max(axis=1)

    # Broadcast min_ and max_ array to the shape of data
    min_ = np.stack([np.tile(x, (frame_width, 1)) for x in min_])
    max_ = np.stack([np.tile(x, (frame_width, 1)) for x in max_])

    # Return normalized data
    return (data - min_) / (max_ - min_)



def load_data(data_path, shape):
    
    # Import data
    data = import_data(data_path=data_path, shape=shape)
    # Divide in Train, Validation and Test sets and separately scale data
    x_train, x_val, x_test = [min_max_scale(set_) for set_ in train_val_test_split(data)]

    # Split train set in subsequent windows
    x_train = split(x_train, width=15, stride=1)
    print('Train set shape \t', x_train.shape)

    # Split validation set in subsequent windows
    x_val = split(x_val, width=15, stride=1)
    print('Validation set shape \t', x_val.shape)
    
    print('Number of test tracks \t', len(x_test))
    
    return x_train, x_val, x_test

    
def create_model(verbose=True, print_path=None):
    # This is the input placeholder
    input_seq = Input(shape=(14, 3, 1))

    X = Flatten()(input_seq)

    X = Dense(21, activation='tanh')(X)
    X = Dense(7, activation='tanh')(X)

    X = Dense(3, activation='tanh')(X)
    
    prediction = Reshape((1, 3, 1,))(X)

    predictor = Model(input_seq, prediction)

    predictor.compile(optimizer='Nadam', loss= "mean_absolute_error")

    # Print architecture information
    if verbose == True:
        predictor.summary()
    # Save architecture schema on file
    if print_path != None:
        plot_model(autoencoder, show_shapes=True, to_file=print_path, rankdir='LR', show_layer_names=False) #TB o LR
        
    return predictor



def train_model(x_train, x_val, model, max_epochs=250, plot=True):
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    input_train = x_train[:,:14,:].reshape(x_train.shape[0], 14, 3, 1)
    output_train = x_train[:,14,:].reshape(x_train.shape[0], 1, 3, 1)
    
    input_val = x_val[:,:14,:].reshape(x_val.shape[0], 14, 3, 1)
    output_val = x_val[:,14,:].reshape(x_val.shape[0], 1, 3, 1)
    
    history = model.fit(input_train, output_train,
                    epochs = max_epochs,
                    steps_per_epoch = 100,
                    shuffle = True,
                    batch_size = None,
                    validation_data=(input_val, output_val),
                    callbacks = [es],
                    validation_steps = 1)
    
    if plot:
    © 2020 GitHub, Inc.
    Terms
    Privacy
    Security
    Status
    Help


        
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



def predict_track(data, model, track_index, frame_width):
    """
    Select one track and compute the denoised version using the specified model
    """
    # Select only one track and add a supplementary dimension on the first axis
    data = np.expand_dims(data[track_index], axis=0)
    # Split data adding a new point for each new window
    splitted_data = split(data, width=frame_width, stride=1)
    # Compute the predicted data for each splitted frame
    predicted_data = model.predict(np.expand_dims(splitted_data, axis=3)).squeeze()
    
    return predicted_data.T



def plot_comparison(track_index, var_index, savefig=False):
    """
    Select one track, one component of that track and plot the original track and the denoised one.
    """
    # Computed the predicted track
    predicted = predict_track(data=x_test, model=predictor, track_index=track_index, frame_width=14)
    # Pad undefined values with nans
    predicted = np.pad(predicted, ((0,0),(13,0)), 'constant', constant_values=np.nan)

    # Plot the predicted serie alongside the original one
    fig = plt.figure(figsize=(18,6))
    plt.scatter(range(predicted.shape[1]), predicted[var_index], marker='.')
    plt.scatter(range(x_test.shape[1]), x_test[track_index,:,var_index], marker='+')
    
    # plt.xlim(70, 110)
    
    plt.legend(('predicted', 'true'))
    plt.grid()
    
    if savefig:
        plt.savefig('dim{}.png'.format(INDEX))
    plt.show()
    
    return None
    
    

main():
    # Load dataset
    x_train, x_val, x_test = load_data(data_path=DATA_PATH, shape=SHAPE)
    # Create or refresh the neural network
    predictor = create_model()
    # Train the net
    history = train_model(x_train, x_val, model=predictor, max_epochs=250)
    # Save the model
    with read(MODEL_PATH, 'wb') as infile:
        pkl.dump(predictor, infile)
    
    
    
if __name__ == "__main__":
    main()
    
    
    
plot_comparison(track_index=4, var_index=1)