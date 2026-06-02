# next_host.py
# https://chatgpt.com/share/6982b264-c038-8002-9e97-4f2034966480

# Which professor is most likely to be the next host of Survivor? To answer this question, project each 
# professor into the reduced "Survivor face space" and apply nearest neighbor classification to see who 
# looks most similar to Jeff Probst.


import argparse
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.util import montage
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pickle as pk


ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
PROFS = os.path.join(ROOT, 'data/professors/')


def main(PROFS):
    print("=================================")
    print("Project 1 - Finding the Next Host")
    print("=================================")

    # Load data
    surWeights = np.load('pca_weights.npz')['arr_0']
    with open('saved_model.pkl', 'rb') as file:
        model = pk.load(file)
    
    profs, plabels = load_images(PROFS)

    # Retrieve Jeff Probst data to use for NN
    print("Retrieving Jeff Probst's weight(s) from the data")
    Jeff = surWeights[0][:model.n_components_]

    # Convert profs data to np array to be usable
    profs = np.array(profs)

    # Transpose our professor data
    profs = profs.T

    # Project the professor data into "survivor face space"
    print("Projecting professor's onto Survivor Space")
    weightsProfs = model.transform(profs)

    # Train the NN algorithm projected Professor data
    print("Set up the Nearest Neighbors algorithm")
    prof_Jeff_NN = NearestNeighbors(n_neighbors=5)
    prof_Jeff_NN.fit(weightsProfs)

    # Query the trained model for Jeff Probst to see which is closest
    dist, nextHostIndex = prof_Jeff_NN.kneighbors(Jeff.reshape(1, -1))
    nextHost = weightsProfs[nextHostIndex.flatten()[0]]

    # Prepare montage of Jeff and the next Host
    reconJeff = model.mean_ + Jeff @ model.components_
    reconNext = model.mean_ + nextHost @ model.components_

    montageTitle = "Reconstructed Faces of Jeff Probst and Next Host (" + str(plabels[nextHostIndex.flatten()[0]]) + ")"
    showThis = np.array([reconJeff, reconNext]).T
    show_montage(showThis, title=montageTitle)



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
    main(PROFS)