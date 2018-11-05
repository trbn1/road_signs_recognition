# -*- coding: utf-8 -*-

import display_info as di
import model as m

from load_data import load_data
from preprocess import preprocess
from sklearn.metrics import classification_report


# Load training and testing datasets.
train_data_dir = 'BelgiumTS/Training'
#train_data_dir = 'GTSRB/Final_Training/Images'
X_train, X_test, y_train, y_test = load_data(train_data_dir)
# Print number of images.
print('Unique training labels: {0}\nTotal training images: {1}'.format(len(set(y_train)), len(X_train)))
print('Unique test labels: {0}\nTotal training images: {1}'.format(len(set(y_test)), len(X_test)))
# Print images and labels info
#di.display_images_and_labels(X_train, y_train, 'hsv')
#di.display_label_images(X_train, 32, 'hsv', y_train)
#di.display_label_images(X_test, 32, 'hsv', y_test)

# Preprocess images.
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Train model.
model = m.train_model(X_train, y_train)
print('Model parameters:\n', model)
# Predictions
predictions = m.predict(model, X_test)
print('Classification report:\n', classification_report(y_test, predictions))