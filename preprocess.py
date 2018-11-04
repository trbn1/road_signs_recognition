# -*- coding: utf-8 -*-

"""
This is a script used for preprocessing images.
"""
from numpy import reshape
from skimage import color, transform


def preprocess(images):
    """Convert images to grayscale, then resize them to 32x32
    and reshape array dimension from 3D to 2D"""
    # Convert images to grayscale.
    images = [color.rgb2gray(image)
                for image in images]

    # Resize images to 32x32.
    images32 = [transform.resize(image, (32, 32), mode='constant')
                for image in images]

    # 3D to 2D.
    images32_size = len(images32)
    return reshape(images32, (images32_size, -1))