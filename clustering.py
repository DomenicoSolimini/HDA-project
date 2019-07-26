# %%

import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans
import numpy as np


# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## K-means ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
n_clusters_ = 2
def k_means(data):
    """ Performs a k-means clustering and plot the obtained result
    """
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters_, n_init=10)
    output = kmeans.fit(data)
    plt.subplot(1,1,1)
    plt.title('Clusters identified using K-means: %d' % n_clusters_)
    plt.scatter(data[:, 0], data[:, 1], c=output.labels_)
    plt.show()



def k_means_preproc(image, n_clust):
    """Takes as input an image and the number of colors to keep, returns
    a segmented image with number of colors equals to n_clust, this is
    a vector quantization procedure done through k-means algorithm
    """
    X = image #.reshape(-1, 4)
    kmeans = KMeans(n_clusters=n_clust).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(image.shape)



def plot_result(image, segmented_image):
    """Plot the result of k_means_preproc
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(image)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(segmented_image)
    ax[1].set_title('2-color Image', size=16)
    plt.show()



# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## DBSCAN ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


def dbscan(data, eps=50, min_samples=10):
    """ Perform a DBSCAN clustering.
    """
    model = DBSCAN(eps, min_samples)
    pred = model.fit(data)

    core_samples_mask = np.zeros_like(pred.labels_, dtype=bool)
    core_samples_mask[pred.core_sample_indices_] = True
    labels = pred.labels_

    return pred



def ellix_param(pred, data):
    """
    """
    labels = pred.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    centre, var = [], []
    for i in range(n_clusters):
        num = len(pred.labels_[pred.labels_ == i])
        centre.append(data[:num, :2].mean(axis=0))
        var.append(np.sqrt(data[0:num, :2].var(axis=0)))
        data = data[num:]
    return centre, var



def dbscan_info(pred):
    """ Print basic info about the clustering.
    """
    labels = pred.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)



def plot_dbscan_2d(data, model):
    """Plot the discovered clusters with different colors."""
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], c=model.labels_, s=300)
    plt.show()



def plot_dbscan_3d(data, model):
    """Plot the discovered clusters with different colors, showing both
    the points coordinates and the intensity of the signal.
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=300)
    ax.view_init(azim=200)
    plt.show()


# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Gaussian Mixture ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import mixture
from matplotlib.patches import Ellipse

# %%
#choose here the number of cluster (2, one is the subject, the other is noise)
#or maybe there are particular case to investigate...


def gaussian_mixtures(data, n_cluster_=2, Verbose=True, Bayesian=False):
    """ Perform a Gaussian Mixture and a Bayesian Gaussian Mixture clustering
    """
    if Bayesian:
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_clusters_, covariance_type='full').fit(data)
        labels = dpgmm.predict(data)
        means = dpgmm.means_

        if Verbose:
            print("The estimated mean vectors for bgmm are:\n {} \n".format(dpgmm.means_))
            print("The estimated covariance matrices for bgmm are:\n {} ".format(dpgmm.covariances_))
    else:
        gmm = mixture.GaussianMixture(n_components = n_clusters_, covariance_type="full").fit(data)
        labels = gmm.predict(data)
        means = gmm.means_

        if Verbose:
            print("The estimated mean vectors for gmm are:\n {} \n".format(gmm.means_))
            print("The estimated covariance matrices for gmm are:\n {}\n ".format(gmm.covariances_))

    return(labels, means)

# %%



def plot_mixture(gmm):
    """ Plot the results of the Gaussian Mixture model, the bayesian gmm
    and the default image
    """
    labels, dlabels = gmm

    plt.figure(figsize=(15, 6))

    plt.subplot(1,3,1)
    plt.title("Default Image")
    plt.scatter(data[:,0], data[:,1])

    plt.subplot(1,3,2)
    plt.title('Clusters from gmm: %d' % n_clusters_)
    plt.scatter(data[:, 0], data[:, 1], c=labels)

    plt.subplot(1,3,3)
    plt.title('Clusters from bgmm: %d' % n_clusters_)
    plt.scatter(data[:, 0], data[:, 1], c=dlabels)


    plt.tight_layout()
    plt.show()




# %%
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))




def plot_gmm(gmm, X, labels, label=True, ax=None):
    """ apply draw_ellipse to gmm output
    """
    ax = ax or plt.gca()
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)




def plot_with_el(gmm, dpgmm):
    """ plot the gmm and dpgmm figure with ellipses
    """
    plt.figure(figsize=(13, 10))
    plt.subplot(2,1,1)
    plt.title('Gaussian Mixture with ellipses')
    plot_gmm(gmm, data, labels)
    plt.subplot(2,1,2)

    plt.title('Bayesian Gaussian Mixture with ellipses')
    plot_gmm(dpgmm, data, dlabels)
    plt.tight_layout()
    plt.show()
