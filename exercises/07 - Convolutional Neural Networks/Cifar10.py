# -*- coding: utf-8 -*-
"""
Updated 2019.10.28

@author: Kristian Hovde Liland
"""

"""
EXERCISE: Cifar10 classification
Create a CNN using the recipe below and test it on the Cifar10 dataset.

1. Convolutional input layer, 32 feature maps with a size of 3 x 3, a rectifi
er activation
    function and a weight constraint of max norm set to 3.
2. Dropout set to 20%.
3. Convolutional layer, 32 feature maps with a size of 3 x 3, a recti
er activation function
    and a weight constraint of max norm set to 3.
4. Max Pool layer with the size 2 x 2.
5. Flatten layer.
6. Fully connected layer with 512 units and a recti
er activation function.
7. Dropout set to 50%.
8. Fully connected output layer with 10 units and a softmax activation function.
"""

# Import data
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Create model

# Run model

# Evalutate model

