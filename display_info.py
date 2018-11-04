# -*- coding: utf-8 -*-

"""
This is a script used for displaying dataset images info.
"""
import matplotlib.pyplot as plt


def display_images_and_labels(images, labels, image_cmap):
    """Display the first image of each label."""
    cmap = image_cmap
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

def display_label_images(images, label, image_cmap, label_type):
    """Display images of a specific label."""
    cmap = image_cmap
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = label_type.index(label)
    end = start + label_type.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image, cmap)
    plt.show()