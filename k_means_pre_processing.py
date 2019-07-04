# da mettere subito dopo importing data

####

from matplotlib.image import imread
from sklearn.cluster import KMeans
image = imread("prova.png")

def k_means_preproc(image, n_clust):
    """Takes as input an image and the number of colors to keep, returns
    a segmented image with number of colors equals to n_clust, this is 
    a vector quantization procedure done through k-means algorithm
    """
    X = image.reshape(-1, 4)
    kmeans = KMeans(n_clusters=n_clust).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(image.shape)
    
def plot_result(image, segmented_image):
    """Plot the result of k_means_preproc
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(image)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(segmented_image)
    ax[1].set_title('2-color Image', size=16)
    plt.show()
  
#%%

segmented_img = k_means_preproc(image, 2)
plot_result(image, segmented_img)
