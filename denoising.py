import numpy as np



def remove_line(img_bw, axis=0, tickness=0):
    """ Remove the central line from a properly formatted figure. The image
    should have only black and white pixel, the line is expected to be white
    with a black background. If 'axis' == 0, it removes vertical lines, if
    'axis' == 1, it removes orizontal lines (if any). All the pixels whose
    distance from the line is less or equal then 'tickness' are removed as well.
    """
    #percentage of active pixel over the whom the column is considered as full
    threshold = 0.5*img_bw.shape[axis]
    # indices of columns to be cleaned
    ind = [i for i in range(img_bw.shape[1-axis]) if np.sum(img_bw[:,i])>threshold]
    # increases the tickness of the line
    for i in ind:
        for n_pix in range(tickness):
            ind.append(i+n_pix)
            ind.append(i-n_pix)
    ind = list(set(ind)) #removes duplicates
    img_bw[:, ind] = np.zeros((img_bw.shape[axis], len(ind)))
    return img_bw



def select_coord(img_bw):
    """ Stores the coordinates of all the white points contained
    in an array.
    """
    return np.array([[i, j] for i in range(img_bw.shape[0])
            for j in range(img_bw.shape[1]) if img_bw[i,j] == 1])



def select_points(img_bw, img_gray):
    """ Stores the coordinates of all the white points contained
    in an array plus the gray level in the associated gray scale
    image.
    """
    return np.array([[i, j, img_gray[i,j]] for i in range(img_bw.shape[0])
            for j in range(img_bw.shape[1]) if img_bw[i,j] == 1])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Denoising ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def denoisingCV(img_name):
#     ''' Takes as input the name of the file where the image is stored
#     '''
#     import cv2
#
#     img = cv2.imread(img_name)
#     b,g,r = cv2.split(img)           # get b,g,r
#     rgb_img = cv2.merge([r,g,b])     # switch it to rgb
#
#     dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 21, 50)
#     b,g,r = cv2.split(dst)           # get b,g,r
#     rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
#     return rgb_dst
#
#
#
# def denoisingPCA(img, n):
#     ''' Takes as imput an image and an integer that represents the number
#     of principal components to consider and returns the denoised image.
#     '''
#     from sklearn.decomposition import PCA
#
#     #img = mpimg.imread(img_name)
#     dim = img.shape
#     #img_r = np.reshape(img, (dim[0], dim[1]*4))
#
#     img_r = img
#     ipca = PCA(n, svd_solver='randomized').fit(img_r)
#     img_c = ipca.transform(img_r)
#
#     print('Explained variance ratio: ',np.sum(ipca.explained_variance_ratio_))
#     # To visualize how PCA has performed this compression, let's inverse
#     # transform the PCA output and reshape for visualization using imshow
#     temp = ipca.inverse_transform(img_c)
#     #reshaping to original size
#     temp = np.reshape(temp, dim)
#
#     return temp
#

# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Pre-Clustering ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# def rgb2gray(rgb):
#     """ Convert a colored image into a gray scale image
#     """
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
#
#
# def rgb2bw(rgb):
#     """ Convert a colored image into an image containing only black and white
#     pixels, all the gray shades are removed. Note that the function doesn't
#     work taking in input a grayscale image.
#     """
#     bw = rgb2gray(rgb)
#     m = bw.mean()
#     bw[bw >= m] = 1; bw[bw < m] = 0
#     return bw
#


# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Plotting ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def plotdenoised(img1, img2):
#     """ Plot 2 images side by side
#     """
#     fig = plt.figure(figsize=[8,6])
#     ax = fig.add_subplot(121)
#     ax.imshow(img1)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax = fig.add_subplot(122)
#     ax.imshow(img2)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     plt.show()


#
# def plot_noaxbw(img):
#     """ plot an image without white borders and axis
#     """
#     fig = plt.figure(figsize=[6,6])
#     ax = fig.add_subplot(111)
#     ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#
