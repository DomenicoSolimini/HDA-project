#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Gaussian Mixture ########
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import mixture
from matplotlib.patches import Ellipse

#%%
# choose here the number of cluster (2, one is the subject, the other is noise)
# or maybe there are particular case to investigate...
n_clusters_ = 2

def gaussian_mixtures(data):
    """ Perform a Gaussian Mixture and a Bayesian Gaussian Mixture clustering
    """
    gmm = mixture.GaussianMixture(n_components = n_clusters_,covariance_type="full").fit(data)
    labels = gmm.predict(data)

    print("The estimated mean vectors for gmm are:\n {} \n".format(gmm.means_))
    print("The estimated covariance matrices for gmm are:\n {}\n ".format(gmm.covariances_))


    dpgmm = mixture.BayesianGaussianMixture(n_components=n_clusters_,
                                        covariance_type='full').fit(data)
    dlabels = dpgmm.predict(data)

    print("The estimated mean vectors for bgmm are:\n {} \n".format(dpgmm.means_))
    print("The estimated covariance matrices for bgmm are:\n {} ".format(dpgmm.covariances_))
  
    return(labels, dlabels)


#%%
def plot_mixture(gmm):
    """ Plot the results of the Gaussian Mixture model, the bayesian gmm
    and the default image
    """
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


#%%
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


#%%
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
  

#%%
gaussian_mixtures(data)

plot_mixture(gmm)

plot_with_el(gmm, dpgmm)   

k_means(data)

