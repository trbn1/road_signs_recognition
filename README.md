# Road Signs Recognition

Road Signs Recognition using multilayer perceptron neural network algorithm

## Dependencies

Following software was tested and executed using Anaconda 5.3.0 on Windows.

*get_model.py* expects image files with road signs under *dataset* directory

During develompent phase we are using Belgian Traffic Sign Dataset linked in references, specifically:

* BelgiumTSC_Training (171.3MBytes)
* BelgiumTSC_Testing (76.5MBytes)

Example format after unpacking:

```
datasets/BelgiumTS/Training/
datasets/BelgiumTS/Testing/
```

## Instructions

* Install Anaconda
* Configure your software to use Anaconda enviornment
* Unpack dataset to project directory as explained earlier
* Cd to project directory

To train the classifier:
```
python get_model.py
```

To classify a single image, using a pre-trained classifier:
```
python classify.py
```

To start a graphical interface for classifying images:
```
python classify_gui.py
```

To run road signs classifying accuracy tests with added noise or blur (parameters configured inside .py file):
```
python distort_and_classify.py
```

## References

[Belgian Traffic Sign Dataset - BelgiumTS](https://btsd.ethz.ch/shareddata/)

[Reference Belgian Traffic Signs - OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Road_signs_in_Belgium)

[German Traffic Sign Recognition Dataset - GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)

[Traffic Sign Recognition with TensorFlow by Waleed Adbulla](https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6)