import matplotlib.pyplot as plt
import h5py
import numpy as np
import process
import argparse

hdf = h5py.File("target3_126.hdf5",'r')
x = np.array(hdf['radar'])
n_frame = x.shape[0]

ind = np.random.randint(0, n_frame)
rda = process.range_doppler(x[ind])
ind
plt.matshow(rda)
