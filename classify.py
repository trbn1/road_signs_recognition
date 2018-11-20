# -*- coding: utf-8 -*-

"""
This is a script used for classifying given image.
"""
from sklearn.externals import joblib

from load_data import load_image
from preprocess import preprocess_single


def classify():
    """Classify given image file and return detected sign name."""
    # Load model from file.
    model = joblib.load(MODEL_LOCATION)

    # Load and preprocess the image.
    image = preprocess_single(load_image(IMAGE_LOCATION))

    # Predict road sign.
    prediction = model.predict(image)
    print('\nLabel of predicted road sign: ', prediction[0])

    # Load available road signs names.
    with open(RESULTS_LOCATION) as file_name:
        results = file_name.readlines()
    results = [x.strip() for x in results]

    # Print road sign name.
    end = len(results)
    for i in range(end):
        if prediction[0] == i:
            result = results[i]
    print('Name of predicted road sign: ', result)


# Model file location.
MODEL_LOCATION = 'models/predict_signs_model_50.pkl'
# Test image file location.
IMAGE_LOCATION = 'test_images/1.png'
# Detected signs names file location.
RESULTS_LOCATION = 'test_images/results.txt'


if __name__ == '__main__':
    classify()
