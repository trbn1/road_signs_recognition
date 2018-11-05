# Road Signs Recognition

Road Signs Recognition using multilayer perceptron neural network algorithm (WIP)

## Dependencies

Following software was tested and executed using Anaconda 5.3.0 on Windows.

*road_signs_recognition.py* expects image files with road signs under *dataset* directory

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
* Execute *python main.py*

## References

[Belgian Traffic Sign Dataset - BelgiumTS](https://btsd.ethz.ch/shareddata/)

[German Traffic Sign Recognition Dataset - GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)

[Traffic Sign Recognition with TensorFlow by Waleed Adbulla](https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6)