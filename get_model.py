# -*- coding: utf-8 -*-

"""
This is a script used for training chosen classifier, using a given dataset.
"""
from sklearn.externals import joblib
from sklearn.metrics import classification_report

import model as m

from load_data import load_data
from preprocess import preprocess


def get_model():
    """Load given dataset, process it and train the classifier."""
    # Load training and testing datasets.
    x_train, x_test, y_train, y_test = load_data(TRAIN_DATA_DIR)

    # Print number of images.
    print('Unique training labels: {0}\nTotal training images: {1}'\
            .format(len(set(y_train)), len(x_train)))
    print('Unique test labels: {0}\nTotal test images: {1}'\
            .format(len(set(y_test)), len(x_test)))

    # Preprocess images.
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    # Train model.
    model = m.train_model(x_train, y_train)
    print('Model parameters:\n', model)
    # Export model to file.
    joblib.dump(model, 'models/predict_signs_model.pkl', compress=9)
    # Display predictions accuracy.
    predictions = m.predict(model, x_test)
    print('Classification report:\n', classification_report(y_test, predictions))


# Dataset location.
TRAIN_DATA_DIR = 'datasets/BelgiumTS/Training'
#TRAIN_DATA_DIR = 'GTSRB/Final_Training/Images'


if __name__ == '__main__':
    get_model()
