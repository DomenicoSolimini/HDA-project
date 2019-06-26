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
plt.savefig('prova.png', dpi=400, bbox_inches='tight',pad_inches=-0.01)


# Denoising per immagini a colori
import cv2

img = cv2.imread('prova.png')
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb


dst = cv2.fastNlMeansDenoisingColored(img, None, 50, 50, 7, 21)

b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb

plt.figure(figsize=[10,6])
plt.subplot(121),plt.imshow(rgb_img)
plt.subplot(122),plt.imshow(rgb_dst)
plt.show()
plt.show()
