# -*- coding: utf-8 -*-

"""
This is a script used for classifying given image.
"""
from sklearn.externals import joblib

from load_data import load_image
from preprocess import preprocess_single


def classify(model_location, image_location, results_location, print_result=False, array=False):
    """Classify given image file and return detected sign name."""
    # Load model from file.
    model = joblib.load(model_location)

    # Load and preprocess the image.
    if array is False:
        image = preprocess_single(load_image(image_location))
    else:
        image = preprocess_single(image_location)

    # Predict road sign.
    prediction = model.predict(image)

    # Load available road signs names.
    with open(results_location) as file_name:
        results = file_name.readlines()
    results = [x.strip() for x in results]

    # Print road sign name.
    end = len(results)
    for i in range(end):
        if prediction[0] == i:
            result = results[i]
            label = i

    if print_result:
        print('\nLabel of predicted road sign: ', prediction[0])
        print('Name of predicted road sign: ', result)
    return result, label


# Model file location.
MODEL_LOCATION = 'models/predict_signs_model_50.pkl'
# Test image file location.
IMAGE_LOCATION = 'test_images/1.png'
# Detected signs names file location.
RESULTS_LOCATION = 'reference_images/results_en.txt'


if __name__ == '__main__':
    classify(MODEL_LOCATION, IMAGE_LOCATION, RESULTS_LOCATION, True)
