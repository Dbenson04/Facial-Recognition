# professor_recon.py

# Which professor looks least like a face according to the underlying facial 
# features in the Survivor dataset? To answer this question, reconstruct each 
# professor's face using the limited number of principal components from (1), 
# then compute the Euclidean distance from the reconstructed face to the 
# original. Largest distance indicates least likely to be a "face".

import argparse
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA
from skimage.util import montage
from sklearn.preprocessing import MinMaxScaler
import pickle as pk

ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
DATAFOLDER = os.path.join(ROOT, "data/survivor/")
TESTFOLDER = os.path.join(ROOT, "data/professors/")

parser = argparse.ArgumentParser(description="Apply our trained PCA model to reconstruct professor's faces and determine who looks least like a face")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')


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
 
def load_images(directory, verbose=True):
    """Load data (and labels) from a directory."""
    print(f"Loading files from: {directory}")
    files = os.listdir(directory)
    num_files = len(files)
    
    images = [] # initialize array of images
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\tProcessing file: {file}")
            img = imread(os.path.join(directory, file))
            
            if img.ndim == 3: # convert to grayscale
                img = img.mean(axis=2)
                
            img_vector = img.astype(np.float64).flatten() # convert image to a vector after ensuring in float64 type
            images.append(img_vector)
    data = np.column_stack(images) # convert all images to numpy array -- each image is it's own column in array
        
    return data


def main(args, DATAFOLDER, TESTFOLDER):
    print("============================")
    print("Project 1 - Face Recognition")
    print("============================")
    
    # Load our saved PCA model
    modelDir = os.path.join(ROOT, "saved_model.pkl")
    with open(modelDir, 'rb') as file:
        pca = pk.load(file)
    
    # Load testing data (professors) and transform
    test_data = load_images(TESTFOLDER)
    t_data = test_data.T
    test_pca = pca.transform(t_data)
    
    # Reconstruct images using scikit-learn PCA library (inverse.transform())
    recon_test = pca.inverse_transform(test_pca)
    
    # Scale the reconstructed images so they can be visually displayed correctly in a montage format
    normalized_test_data = MinMaxScaler().fit_transform(recon_test.T).T
    
    # Calculate the euclidean distances from each professor's reconstructed face to their original
    largest_distance = float('-inf')
    loser_professor = -1
    for i in range(len(normalized_test_data)):
        x1 = test_data[i] # Original face
        x2 = normalized_test_data.T[i] # Reconstructed face
        dist = np.linalg.norm(x1-x2) # Euclidean distance formula
        if dist > largest_distance:
            largest_distance = dist
            loser_professor = i
            
    # Find the name of the professor to the largest distance from their reconstruction
    n = -1
    for professor in os.listdir(TESTFOLDER):
        n += 1
        if n == loser_professor:
            loser_professor = os.path.splitext(professor)[0]
            break

    # Visually display our results and print them in the terminal
    show_montage(normalized_test_data.T, image_shape=(70, 70), grid_shape=(1,5))
    print(f"The professor who looks least like a face is professor {loser_professor} with a distance of {largest_distance}")
    
        
if __name__ == "__main__":
	main(parser.parse_args(), DATAFOLDER, TESTFOLDER)