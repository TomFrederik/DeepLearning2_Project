import numpy as np
import pandas as pd
import sys
import os
import argparse

def join_datasets(folder):
    """This function implements the functionality described in the comments above"""
    
    # Get the list of folders
    folders = os.listdir(folder)
    folders = [int(f) for f in folders]
    folders.sort()
    
    # Get the list of labels
    labels = []
    for f in folders:
        label_path = folder + '/' + str(f) + '/labels.csv'
        label_df = pd.read_csv(label_path)
        labels.append(label_df)
    
    # Concatenate the labels
    labels = pd.concat(labels)
    
    # update the index to reflext the concatenation
    labels.index = range(len(labels))
    
    # change name of each image to {index}.jpg
    labels['im_name'] = labels.index.map(lambda x: str(x) + '.jpg')

    # Get the list of images
    images = []
    for f in folders:
        image_path = folder + '/' + str(f) + '/ims'
        image_files = os.listdir(image_path)
        for image_file in image_files:
            image_path = folder + '/' + str(f) + '/ims/' + image_file
            images.append(image_path)
    
    # Copy the images to a single directory
    os.makedirs(folder + '/all')
    for i, image_path in enumerate(images):
        image_file = f"{i}.jpg"
        os.system('cp ' + image_path + ' ' + folder + '/ims/' + image_file)
    
    # Save the labels
    labels.to_csv(folder + '/labels.csv')
    
    # Now delete all the other folders
    for f in folders:
        os.system('rm -rf ' + folder + '/' + str(f))
    
    print('Done!')


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()
    
    folder = args.folder
    
    join_datasets(folder)
    