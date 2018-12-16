# -*- coding: utf-8 -*-

"""
This is a script used for distorting and classifying given image.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#from imageio import imwrite
from skimage.filters import gaussian
from skimage.util import random_noise

from classify import classify
from load_data import load_image


def add_noise(curr_image, noise_amount):
    """Add random noise to given image and return it."""

    return random_noise(curr_image, mode='s&p', amount=noise_amount)


def add_blur(curr_image, blur_amount):
    """Add blur to given image and return it."""

    return gaussian(curr_image, multichannel=True, sigma=blur_amount)


# Model file location.
MODEL_LOCATION = 'models/predict_signs_model_50.pkl'
# Test image file location.
IMAGE_LOCATION = 'test_images/1.png'
# Detected signs names file location.
RESULTS_LOCATION = 'reference_images/results_en.txt'

# Current image label.
IMAGE_LABEL = 30
# Noise iterations.
NOISE_ITERATIONS = 10000

NOISE = 0

# Blur iterations.
BLUR_ITERATIONS = 10000
MAX_BLUR = BLUR_ITERATIONS / 100


if __name__ == '__main__':
    LABELS = []
    ITERATIONS = []
    ERRORS = 0
    FIRST_ERROR = 0

    if NOISE is 1:
        # Noise loop.
        for i in range(NOISE_ITERATIONS):
            DISTORTED_IMAGE = add_noise(load_image(IMAGE_LOCATION), i / NOISE_ITERATIONS)
            #imwrite('distorted' + str(i) + '.png', DISTORTED_IMAGE)
            #print('Current noise amount: ', i / (NOISE_ITERATIONS * 2))
            result, label = classify(MODEL_LOCATION, DISTORTED_IMAGE, RESULTS_LOCATION, False, True)
            if label is not IMAGE_LABEL:
                if FIRST_ERROR is 0:
                    FIRST_ERROR = i / NOISE_ITERATIONS
                ERRORS = ERRORS + 1
            LABELS.append(ERRORS)
            ITERATIONS.append(i / NOISE_ITERATIONS)
        if FIRST_ERROR is not 0:
            print('First error: ', FIRST_ERROR)

        # Plot.
        FIG, AX = plt.subplots()
        AX.plot(ITERATIONS, LABELS, color='k')
        AX.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        AX.yaxis.set_major_formatter(mtick.PercentFormatter(NOISE_ITERATIONS))
        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0, top=(ITERATIONS[-1] * NOISE_ITERATIONS))
        plt.title('Amount of incorrectly detected signs\ndepending on the amount of noise in image\n' +
                  'iterations = ' + str(len(ITERATIONS)) + ', label = ' + str(IMAGE_LABEL))
        plt.xlabel('Amount of noise in image [%]')
        plt.ylabel('Amount of incorrectly detected signs [%]')
        plt.tight_layout()
        plt.show()
    else:
        # Blur loop.
        for i in range(BLUR_ITERATIONS):
            BLURRED_IMAGE = add_blur(load_image(IMAGE_LOCATION), (i / MAX_BLUR) / 5)
            #imwrite('tmp/' + str(i) + '.png', BLURRED_IMAGE)
            #print('Current blur amount: ', i + (i / BLUR_ITERATIONS))
            result, label = classify(MODEL_LOCATION, BLURRED_IMAGE, RESULTS_LOCATION, False, True)
            if label is not IMAGE_LABEL:
                if FIRST_ERROR is 0:
                    FIRST_ERROR = i / BLUR_ITERATIONS
                ERRORS = ERRORS + 1
            LABELS.append(ERRORS)
            ITERATIONS.append(i / BLUR_ITERATIONS)
        if FIRST_ERROR is not 0:
            print('First error: ', FIRST_ERROR)

        # Plot.
        FIG, AX = plt.subplots()
        AX.plot(ITERATIONS, LABELS, color='k')
        AX.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        AX.yaxis.set_major_formatter(mtick.PercentFormatter(BLUR_ITERATIONS))
        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0, top=(ITERATIONS[-1] * BLUR_ITERATIONS))
        plt.title('Amount of incorrectly detected signs\ndepending on the amount of blur in image\n' +
                  'iterations = ' + str(len(ITERATIONS)) + ', label = ' + str(IMAGE_LABEL))
        plt.xlabel('Amount of blur in image [%]')
        plt.ylabel('Amount of incorrectly detected signs [%]')
        plt.tight_layout()
        plt.show()
