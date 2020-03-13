This code consists in three main python files:
1. `load_data.py` that load the training tracks and perform some data augmentation;
2. `build_autoencoder.py` that specifies the autoencoder architecture, the way in which the autoencoder is used to predict new points and train the model;
3. `GT-tracks_filtering.py` that test the model on _ground truth data_ and returns the filtered track and the RMSD between the filtered and the true trajectory.

The testing can be performed over a specific track passing the name as parameter using the sintax `python GT-tracks_filtering --track "track_name"`. A shell file is added to automatize the work over the whole testing tracks returning also a file containing the RMSE computed over each track called `metrics.txt`.
