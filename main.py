# -*- coding: utf-8 -*-

from load_data import load_data
from preprocess import preprocess


# Load training and testing datasets.
train_data_dir = 'BelgiumTS/Training'
test_data_dir = 'BelgiumTS/Testing'
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

# Print number of images.
print('Unique training labels: {0}\nTotal training images: {1}'.format(len(set(train_labels)), len(train_images)))
print('Unique test labels: {0}\nTotal training images: {1}'.format(len(set(test_labels)), len(test_images)))

"""# Print images and labels info
display_images_and_labels(images, labels, "hsv")
display_label_images(images, 32, "hsv")
# Print size of first 5 images.
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
display_images_and_labels(images, labels, "gray")
display_images_and_labels(images32, labels, "gray")
# Print size of first 5 resized images.
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))"""

# Preprocess images
train_images32_2d = preprocess(train_images)
test_images32_2d = preprocess(test_images)