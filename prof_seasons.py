# prof_seasons.py
# https://chatgpt.com/share/69865d96-1c00-8002-98c4-02edf047fc20

# Which season would each professor most likely be on? To answer this question, 
# use k-means clustering on the PCA-reduced Survivor faces, then assign each of 
# the PCA-reduced professor faces to the nearest cluster. The average season of 
# Survivor castaways in the cluster (not including Jeff Probst) is the assigned 
# season for that professor.

import argparse
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.util import montage
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle as pk
from scipy.spatial.distance import cdist
from statistics import mode


ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
PROFS = os.path.join(ROOT, 'data/professors/')

parser = argparse.ArgumentParser(description="Use K-means clustering on the survivor faces, assign each professor to the best cluster.")
parser.add_argument("-k", metavar="clusters", default=5, help='number of clusters to generate (default=5)')
parser.add_argument("--seed", "-s", type=int, default=None, help='random seed for repeatability')
parser.add_argument('--iter', '-i', type=int, default=10, help='maximum number of iterations (default=10)')


def main(args, PROFS):
    
    # Load data
    surWeights = np.load('pca_weights.npz')['arr_0']
    surNames = np.load('pca_weights.npz')['arr_1']
    # each index (0, 1, 2, etc) corresponds to the matching weights-name pair from surWeights and surNames respectively
    with open('saved_model.pkl', 'rb') as file:
        model = pk.load(file)
    
    profs, plabels = load_images(PROFS)

    # Set the seed and number of means
    print("Seeding starting means")
    np.random.seed(args.seed)
    mu = np.random.uniform(-10, 10, size=(args.k, model.n_components_))

    # Use Scikit Learn to run K-Means clustering
    print("Running the K-Means Clustering algorithm")
    kmeans = KMeans(n_clusters=args.k, n_init=args.iter, random_state=0)
    surClusters = kmeans.fit(surWeights).labels_

    # Convert profs data to np array to be usable
    profs = np.array(profs)

    # Transpose our professor data
    profs = profs.T

    # Project the professor data into "survivor face space"
    print("Projecting professor's onto Survivor Space")
    weightsProfs = model.transform(profs)

    # Assign each professor to a cluster
    print("Assigning each professor to a cluster")
    clustersProfs = kmeans.fit(weightsProfs).labels_

    # For each professor, take the mode season of all in their cluster, and assign that season to that professor
    print("Assigning each professor to a season of Survivor")
    modes = sort_survivors(surWeights, surNames, surClusters, args.k)
    for profIndex in range(len(weightsProfs)):
        # Variable cluster is index of corresponding mode for the cluster of this professor
        cluster = clustersProfs[profIndex]
        print("The assigned season for " + str(plabels[profIndex]) + " is Season " + str(modes[cluster]))


def sort_survivors(weights, names, clusterLabels, k):
    """Sorts the survivors by cluster and returns the mode season of each cluster"""
    # Sort the survivors by cluster
    sorted = [[] for _ in range(k)]

    for each in range(len(clusterLabels)):
        # Only record the season of each survivor (not Jeff Probst)
        season = int(names[each][1:3])
        if season != 0:
            # pdb.set_trace()
            sorted[clusterLabels[each]].append(season)
    
    # Create list with mode of each cluster and return
    modes = []

    for clust in sorted:
        modes.append(mode(clust))
    
    return modes


# taken and modified from in-class PCA example
def load_images(dataDir):
    """Load data and labels from directory - Modified slightly from in-class PCA example"""
    print(f"Loading images from {dataDir}")
    files = os.listdir(dataDir)

    images = []
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = imread(os.path.join(dataDir, file))

            if img.ndim == 3:
                img = img.mean(axis=2)
            
            img_vector = img.astype(np.float64).flatten()
            images.append(img_vector)
    data = np.column_stack(images)

    # Extract labels from filenames ( the number of the donut in the file name )
    labels = np.array([str(file.split('.')[0]) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # whole label for column in rankings.csv, index [1, 3] for season #, index [4, :] for name with _ separator

    return data, labels

# Taken without modification from in-class PCA example
def show_montage(x, image_shape=None, grid_shape=None, title=""):
    """Display set of images as montage
    Parameters:
    x : np.array (num_pixels, num_images)
        Array of images to display
    image_shape : tuple (height, width)
        Shape of each image, num_pixels should be equal to height * width
    grid_shape : tuple (num_rows, num_cols)
        (optional) Shape of the montage grid
    title : str
        (optional) Title for the matplotlib figure
    """
    if not image_shape:
        height_and_width = int(x.shape[0] ** (1/2))
        image_shape = (height_and_width, height_and_width)
    
    images = x.T.reshape((-1, *image_shape)) # must transpose so that the pixels arent divided across images

    m = montage(images, grid_shape=grid_shape, fill=0, padding_width=1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show(block=True) # will show montage until tab of images is closed manually by user


if __name__ == "__main__":
    main(parser.parse_args(), PROFS)