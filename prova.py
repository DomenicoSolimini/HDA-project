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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## provo la pca ########
# ad occhio sembra meglio (e più figo) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.image as mpimg 
img = mpimg.imread('prova.png') 
print(img.shape )
plt.figure(figsize=[10,6])
plt.axis('off') 
plt.imshow(img)

img_r = np.reshape(img, (3164, 8516)) # 2129 * 4 = 8516
print(img_r.shape )

from sklearn.decomposition import RandomizedPCA
ipca = RandomizedPCA(20).fit(img_r) 
img_c = ipca.transform(img_r) 
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))
# spiega il 67% dei dati, possiamo giocare con questo parametro,
# con 20 buon risultato secondo me: vedere output


#OK, now to visualize how PCA has performed this compression, 
# let's inverse transform the PCA output and 
#reshape for visualization using imshow.
temp = ipca.inverse_transform(img_c)
print(temp.shape)

#reshaping to original size
temp = np.reshape(temp, (3164, 2129, 4))
print(temp.shape)

# output
plt.figure(figsize=[10,6])
plt.axis('off') 
plt.imshow(temp) 
plt.subplot(121),plt.imshow(rgb_img)
plt.subplot(122),plt.imshow(temp)
