# -*- coding: utf-8 -*-

"""
This is a script used for loading data from files into array.
"""
import os
import skimage.data

from sklearn.model_selection import train_test_split


def load_data(data_dir):
    """Loads a data set and returns two lists:
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    ROOT_PATH = "datasets/"
    data_dir = os.path.join(ROOT_PATH, data_dir)
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith('.ppm')]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return train_test_split(images, labels, random_state=0, stratify=labels)

def load_image(image):
    """Load a single image from file and return it in a Numpy array."""
    return skimage.data.imread(image)