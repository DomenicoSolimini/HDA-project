#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## HDAproject - Denoising and Clustering ########
# Last update July, 1 2019
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import numpy as np
import process
import argparse
import random

random.seed(123)

# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Importing Data ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

plt.savefig('prova.png', dpi=700, bbox_inches='tight',pad_inches=-0.01)

# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Denoising ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def denoisingCV(img_name):
    ''' Takes as input the name of the file where the image is stored
    '''
    import cv2

    img = cv2.imread(img_name)
    b,g,r = cv2.split(img)           # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 21, 50)
    b,g,r = cv2.split(dst)           # get b,g,r
    rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
    return rgb_dst



def denoisingPCA(img, n):
    ''' Takes as imput an image and an integer that represents the number
    of principal components to consider and returns the denoised image.
    '''
    from sklearn.decomposition import PCA

    #img = mpimg.imread(img_name)
    dim = img.shape
    img_r = np.reshape(img, (dim[0], dim[1]*4))

    ipca = PCA(n, svd_solver='randomized').fit(img_r)
    img_c = ipca.transform(img_r)

    print('Explained variance ratio: ', np.sum(ipca.explained_variance_ratio_))
    # To visualize how PCA has performed this compression, let's inverse
    # transform the PCA output and reshape for visualization using imshow
    temp = ipca.inverse_transform(img_c)
    #reshaping to original size
    temp = np.reshape(temp, dim)

    return temp


# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Plotting ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotdenoised(img1, img2):
    """ Plot 2 images side by side
    """
    fig = plt.figure(figsize=[8,6])
    ax = fig.add_subplot(121)
    ax.imshow(img1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(122)
    ax.imshow(img2)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.show()



def plot_noaxbw(img):
    """ plot an image without white borders and axis
    """
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Clustering ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rgb2gray(rgb):
    """ Convert a colored image into a gray scale image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



def rgb2bw(rgb):
    """ Convert a colored image into an image containing only black and white
    pixels, all the gray shades are removed. Note that the function doesn't
    work taking in input a grayscale image.
    """
    bw = rgb2gray(rgb)
    m = bw.mean()
    bw[bw >= m] = 1; bw[bw < m] = 0
    return bw


# NON FUNZIONA!!!
def remove_line(img_bw):
    """ Remove the central line from a properly formatted figure. The image
    should have only black and white pixel, the line is expected to be white
    with a black background.
    """
    #percentage of active pixel over the whom the column is considered as full
    threshold = 0.5*img_bw.shape[0]
    # indices of columns to be cleaned
    ind = [i for i in range(img_bw.shape[1]) if np.sum(img_bw[:,i]) > threshold]
    print(ind)
    img_bw[:, ind] = np.zeros((img_bw.shape[0], len(ind)))
    return img_bw



def select_points(img_bw):
    """ NON EFFICIENTE!! VETTORIZZARE I CICLI FOR
    Stores all the white points coordinates in an array.
    """
    A = np.array([0,0], dtype=int) #needed for inizialitazion - to be removed
    for i in range(img_bw.shape[0]):
        for j in range(img_bw.shape[1]):
            if img_bw[i,j] == 1:
                A = np.vstack((A, np.array([i,j], dtype=int)))
    return A[1:,:]


# %%
# parte su dbscan - funziona ma trova troppi cluster - aggiungere visualizzazione
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# %%

data = A

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data[:,0], data[:,1], data[:,2], s=300)
# ax.view_init(azim=200)
# plt.show()

model = DBSCAN(eps=2.5, min_samples=2)
model.fit_predict(data)
pred = model.fit_predict(data)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=300)
# ax.view_init(azim=200)
# plt.show()

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)

# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Denoising Implemented ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sezione per provare le funzioni scritte

img1 = mpimg.imread('prova.png')
img2 = denoisingCV('prova.png')
plotdenoised(img1, img2)

# %%
img3 = denoisingPCA(img1, 12)
img4 = denoisingPCA(img3, 8)
plotdenoised(img1, img4)

# %%
img5 = img4.copy()
img5[img5 < 0.7] = 0
img5[img5 >= 0.7] = 1
plotdenoised(img1, img5)

img5_gray = rgb2gray(img5)
plot_noaxbw(img5_gray)
img5_gray.shape

img5_bw = rgb2bw(img5)
plot_noaxbw(img5_bw)
((img5_bw != 1) & (img5_bw != 0)).any()
# %%

plot_noaxbw(remove_line(img5_bw))
# %%
