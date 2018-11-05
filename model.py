# -*- coding: utf-8 -*-

"""
This is a script used for training a model using given data
and displaying predictions/evaluating.
"""
from sklearn.neural_network import MLPClassifier


def train_model(images, labels):
    """Train model."""
    mlp = MLPClassifier()
    mlp.fit(images, labels)
    return mlp

def predict(model, test_images):
    """Get predictions."""
    predictions = model.predict(test_images)
    return predictions