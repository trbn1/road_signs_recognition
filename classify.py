# -*- coding: utf-8 -*-

"""
This is a script used for classifying given image.
"""
from load_data import load_image
from preprocess import preprocess_single
from sklearn.externals import joblib


# Load model from file.
model = joblib.load('models/predict_signs_model_50.pkl')

# Test image file location.
image = 'test_images/1.png'
results_location = 'test_images/results.txt'

# Load and preprocess the image.
image = preprocess_single(load_image(image))

# Predict road sign.
prediction = model.predict(image)
print('\nLabel of predicted road sign: ', prediction[0])

# Load available road signs names.
with open(results_location) as f:
    results = f.readlines()
results = [x.strip() for x in results] 

# Print road sign name.
END = 62
for i in range(END):
    if prediction[0] == i:
        result = results[i]
print('Name of predicted road sign: ', result)