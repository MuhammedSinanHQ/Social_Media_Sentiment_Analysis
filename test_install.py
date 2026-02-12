import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import sklearn

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("NLTK:", nltk.__version__)
print("Scikit-Learn:", sklearn.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
