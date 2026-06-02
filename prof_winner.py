# prof_winner.py

# Predict which professor is most likely to win the next season of Survivor
# To do this, we took the mean average weights of all previous winners of Survivor
# Then, each professor is compared to the winner's mu via the euclidean distance formula 
# Whichever professor has the shortest distance (smallest value) is predicted to be the next winner

import pandas as pd
import pickle as pk
import os
import pdb
import argparse
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.util import montage

ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
DATAFOLDER = os.path.join(ROOT, "data/survivor/")
TESTFOLDER = os.path.join(ROOT, "data/professors/")


parser = argparse.ArgumentParser(description="Apply a trained PCA model and weights to determine which professor will win the next season of Survivor")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')

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

def main(args, DATAFOLDER, TESTFOLDER):

    # Read the rankings.csv file - then filter it to only show the winners from each season
    df = pd.read_csv("data/survivor/rankings.csv", header=None)
    winners_df = df[df.iloc[:, 1] == 1]

    # Load survivor weights from PCA
    surWeights = np.load('pca_weights.npz')['arr_0']
    modelDir = os.path.join(ROOT, "saved_model.pkl")
    with open(modelDir, 'rb') as file:
        pca = pk.load(file)
        
    # Load test data (professors)
    test_data = load_images(TESTFOLDER)
    t_data = test_data.T
    t_data = pca.transform(t_data)
    
    # Take the mean (mu) of all the winners's weights
    winner_weights = []
    for survivor in os.listdir(DATAFOLDER):
        if survivor.lower().endswith(('.png', '.jpg', '.jpeg')): # Ignores the .csv file in /survivor/
            for winner in winners_df[0].values:
                if winner == os.path.splitext(survivor)[0]:
                    winner_weights.append(surWeights[os.listdir(DATAFOLDER).index(survivor)])
                    
    winner_weights = np.array(winner_weights)
    mu = np.mean(winner_weights, axis=0)

    # Use the euclidean distance formula to predict which professor is closest to the mean average winner (shortest distance)
    best_odds = float("inf")
    winning_professor = -1
    current_professor = -1
    for professor in t_data:
        current_professor += 1
        win_chance = np.linalg.norm(professor - mu)
        if win_chance < best_odds:
            best_odds = win_chance
            winning_professor = current_professor
    
    # Iterate through professors to get name of who was predicted
    n = -1
    for professor in os.listdir(TESTFOLDER):
        n += 1
        if n == winning_professor:
            winner = os.path.splitext(professor)[0]
            break
    
    # Print final result
    print(f"The predicted winner of the next season of survivor is professor {winner.strip("_")}!")
    
    # Display for final results
    winner_eigenface = mu @ pca.components_ + pca.mean_
    prof_eigenface = t_data[winning_professor] @ pca.components_ + pca.mean_
    prof_face = test_data[:, winning_professor]
    
    # Montage the mean average winner's face with the predicted professor's eigenface & original image
    winner_montage = np.column_stack([winner_eigenface, prof_eigenface, prof_face])
    show_montage(winner_montage, image_shape=(70,70), grid_shape=(1,3), title="")

if __name__ in "__main__":
    main(parser.parse_args(), DATAFOLDER, TESTFOLDER)