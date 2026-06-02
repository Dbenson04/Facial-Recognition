# face_recognition.py

# THIS IS FOR TASK 1 ONLY
# Apply PCA to the Survivor faces dataset in order to reduce dimensionality 
# while maintaining at least 90% of the original variance. You are encouraged 
# to use the PCA methods in the scikit-learn library. Save the "model" to 
# file using an appropriate method of your choice (e.g. numpy, pickle, text/csv).

import argparse
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA
from skimage.util import montage
import pickle as pk

ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
DATAFOLDER = os.path.join(ROOT, "data/survivor/")
TESTFOLDER = os.path.join(ROOT, "data/professors/")

parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')

def train(data, labels):
    # Trains PCA model on training data using scikit-learn library
    print(f"Training Model")

    # Perform PCA
    print("Performing PCA")
    E = PCA(n_components=0.9, svd_solver="full")
    E.fit(data)

    # Project data into "survivor face-space" to get our weights
    weights = E.transform(data)

    # Save the model using pickle library, and weights using numpy
    print("Saving Model")
    np.savez('pca_weights.npz', weights, labels)
    with open('saved_model.pkl', 'wb') as file:
        pk.dump(E, file)
    
    # Display montage of our training data (all survivors)
    show_montage(data.T)
    
def show_montage(x, image_shape=None, grid_shape=None, title=""):
	""" Display a set of images as a montage
	
    Parameters
    ----------
    x : np.array (num_pixels, num_images)
    	The input array of images
     image_shape : tuple (height, width)
     	The height and width of each image. Note that num_pixels
      	should equal height * width.
    grid_shape : tuple (rows, cols)
    	(optional) The shape of the montage grid.
    title :
    """
	if not image_shape:
		height_and_width = int(x.shape[0] ** (1/2))
		image_shape = (height_and_width, height_and_width)
        
	images = x.T.reshape((-1, *image_shape)) # (N, H, W)
    
	m = montage(images,
        grid_shape=grid_shape,
        fill=0,
        padding_width=1
    )
	plt.figure(figsize=(8,8))
	plt.imshow(m, cmap='gray')
	plt.axis('off')
	plt.title(title)
	plt.show(block=True)

# taken and modified from in-class PCA example
def load_images(dataDir):
    
    """Load data and labels from directory - Modified slightly from in-class PCA example"""
    print(f"Loading images from {dataDir}")
    files = os.listdir(dataDir)

    images = []
    image_shape = None
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = imread(os.path.join(dataDir, file))

            if img.ndim == 3:
                img = img.mean(axis=2)
            
            if image_shape is None:
                image_shape = img.shape
            
            img_vector = img.astype(np.float64).flatten()
            images.append(img_vector)
    data = np.column_stack(images)

    # Extract labels from filenames ( the number of the donut in the file name )
    labels = np.array([str(file.split('.')[0]) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # whole label for column in rankings.csv, index [1, 3] for season #, index [4, :] for name with _ separator

    return data, labels

def main(args, DATAFOLDER):
    print("============================")
    print("Project 1 - Face Recognition")
    print("============================")
    
    # load training data
    data, labels = load_images(DATAFOLDER)
    data = data.T
   
    train(data, labels)
    
    if args.debug:
        pdb.set_trace()

if __name__ == "__main__":
	main(parser.parse_args(), DATAFOLDER)
