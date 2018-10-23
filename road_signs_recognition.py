import os
import random
import skimage.color
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#
def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
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
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
ROOT_PATH = "datasets/"
train_data_dir = os.path.join(ROOT_PATH, "BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "BelgiumTS/Testing")

images, labels = load_data(train_data_dir)
images_test, labels_test = load_data(test_data_dir)

# Print number of images.
print("Training set:\nUnique Labels: {0}\nTotal Images: {1}\n".format(len(set(labels)), len(images)))
print("Test set:\nUnique Labels: {0}\nTotal Images: {1}\n".format(len(set(labels_test)), len(images_test)))

#
def display_images_and_labels(images, labels, cmap):
    """Display the first image of each label."""
    cmap = cmap
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image, cmap)
    plt.show()

display_images_and_labels(images, labels, "hsv")

#
def display_label_images(images, label, cmap):
    """Display images of a specific label."""
    cmap = cmap
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image, cmap)
    plt.show()

display_label_images(images, 32, "hsv")

# Print size of first 5 images.
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

# Convert images to grayscale
images = [skimage.color.rgb2gray(image)
                for image in images]

images_test = [skimage.color.rgb2gray(image)
                for image in images_test]

display_images_and_labels(images, labels, "gray")

# Resize images to 32x32
images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]

images32_test = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images_test]

display_images_and_labels(images32, labels, "gray")

# Print size of first 5 resized images.
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))