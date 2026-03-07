"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np

def get_data(dataset_choice):
    if dataset_choice == "mnist":
        dataset = keras.datasets.mnist
    elif dataset_choice == "fashion_mnist":
        dataset = keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flattening the image matrices
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Converting to one-hot encoding
    y_train = np.eye(10)[y_train] # 10 is the number of classes in mnist and fashion_mnist datasets
    y_test = np.eye(10)[y_test]

    return (x_train, y_train), (x_test, y_test)
