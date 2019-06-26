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
rda = 20 * np.log10(rda)

# immagine senza assi e sfondo bianco
fig = plt.figure(figsize=[6,6])
ax = fig.add_subplot(111)
ax.matshow(rda)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)

# salva immagine
plt.savefig('prova.png', dpi=400, bbox_inches='tight',pad_inches=-0.1)
