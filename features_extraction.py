import h5py
import numpy as np
import pickle
import scipy
import time
import os
import process
from os import listdir
from os.path import isfile, join
import clustering as clr
import denoising as dns

# %%

def means_vars(CODE, PATH_IN, PATH_OUT):
    """ Performs the entire preprocessing pipeline over a single hdf5 file.
    WARNING: When file dimension is huge, memory error occurs.
    """
    print('Image no.',CODE)
    start_time = time.time()

    # IMPORTING FILE
    hdf = h5py.File(PATH_IN+'target'+CODE+'.hdf5','r')

    x = np.array(hdf['radar'])
    n_frame = x.shape[0]

    r = process.range_doppler(x[0])
    dim = r.shape

    rda = np.empty((n_frame, dim[0], dim[1]))

    for ind in range(n_frame):
        rd = process.range_doppler(x[ind])
        rda[ind, :, :] = np.array(20 * np.log10(rd))
        rd -= np.amax(rd)
        #rd[rd < -45] = -45

    print('RDA:\t\t DONE')

    # DYNAMIC THRESHOLDING
    rda_thresh_g = np.empty((n_frame, dim[0], dim[1]))

    for i in range(n_frame):
        X = rda[0,:,:].flatten()
        m = np.mean(X) + 2*np.std(X)
        rda_thresh_g[i, :, :] = (rda[i,:,:]>m)*1

    print('Thresholding:\t DONE')

    # LINE REMOVING
    rda_bool = np.empty((n_frame, dim[0], dim[1]))

    for i in range(n_frame):
        rda_bool[i,:,:] = dns.remove_line(rda_thresh_g[i,:,:], tickness=0)
        #rda_bool[i,:,:] = remove_line(rda_thresh_g[i,:,:], axis=1, tickness=0)

    print('Line Removing:\t DONE')

    # CLUSTERING
    means_true = np.empty((n_frame, 3))
    # means_noise = np.empty((n_frame, 3))
    vars_true = np.empty((n_frame, 2))

    for ind in range(n_frame):

        data = dns.select_points(rda_bool[ind], rda[ind])
        labels, m = clr.gaussian_mixtures(data, n_cluster_=2, Verbose=False, Bayesian=False)

        i_true = np.argmax(np.array(m)[:, 2])
        means_true[ind, :] = np.array(m)[i_true, :]
        primo = np.std(data[labels==i_true][:,1])
        secondo = np.std(data[labels==i_true][:,2])
        vars_true[ind, :] = np.array([primo, secondo])

    print('Clustering:\t DONE')

    with open(PATH_OUT+'means_vars'+CODE+'.p', 'wb') as outfile:
        pickle.dump((means_true, vars_true), outfile)

    print('Saving Data:\t DONE')

    end_time = time.time()

    print('Total Time: %.2f' % (end_time-start_time),'sec.')

    print('***** Job Done! *****')
    print('')

# %%

# For each file in PATH_IN directory, means_vars() is applied
# every file name needs to be formatted as target+CODE+.hdf5

PATH_IN = 'idrad/train/'
PATH_OUT = 'means_vars_finale/'

# os.system('ls '+ PATH_IN + ' > list.txt')
# f = open("list.txt", "r")
#files = [(f, os.path.getsize(PATH_IN+f)) for f in listdir(PATH_IN) if isfile(join(PATH_IN, f))]
#files = [f for f, size in files if size <= 47236128]

files = listdir(PATH_IN)
codes = []

for line in files:
    codes.append(line[6:-5])

for CODE in codes:
    means_vars(CODE, PATH_IN, PATH_OUT)

print('FINITOOOO!!!')

# creates two files containing all data retrieved so far

PATH_IN = PATH_OUT
PATH_OUT = ''

files = listdir(PATH_IN)
n_files = len(files)

means_array = np.empty((n_files, n_frame, 3))
vars_array = np.empty((n_files, n_frame, 2))

for ind in range(n_files):
    with open(PATH_IN+files[ind], 'rb') as infile:
        means_true, vars_true = pickle.load(infile)

        means_array[ind,:,:] = means_true
        vars_array[ind,:,:] = vars_true


with open(PATH_OUT+'means_array.p', 'wb') as outfile:
        pickle.dump(means_array, outfile)

with open(PATH_OUT+'vars_array.p', 'wb') as outfile:
        pickle.dump(vars_array, outfile)
