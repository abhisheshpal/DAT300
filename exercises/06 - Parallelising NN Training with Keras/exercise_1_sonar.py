# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:02:12 2019

@author: Abhisheshpal
"""


# # After reading this post you will know:
#
# How to wrap Keras models for use in scikit-learn and how to use grid search.
#
# How to grid search common neural network parameters such as learning rate, dropout rate, epochs and number of neurons.
#
# How to define your own hyperparameter tuning experiments on your own projects.
#

"""
1. Train a fully connected neural network in Keras for the sonar data you can 
acquire from 
https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

Info on dataset:
----------------
The dataset describes sonar chirp returns bouncing off different surfaces. The 
input variables are the strength of the returns at different angles. The model 
you will train shall be able to differentiate rocks from metal cylinders.

More info: 
https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)


2. The last column represents the response / target
Rock: R
Metal cylinder: M

3. Play around with the number of layers and activation units in each layer and
compute validation accuracy. Test out different optimisers, number of epochs,
batch sizes. Compare the results. Train the model on a training set and 
validate the performance on a test set.

4. Re-implement model as KerasClassifier using the Keras-Scikit-learn API. Use
scikit-learn API to implement stratified 10-fold cross validation. Compute
average performance across all folds. 

5. Re-implement model in 4. and use it in a pipline with a standard scaler. 
Clock computing times for different hyperparameters in the model.
"""


# ![1_pgTLoLGw0PVaP7ViSyQabA.png](attachment:1_pgTLoLGw0PVaP7ViSyQabA.png)
# #Model Hyperparameters are the properties that govern the entire training process. The below are the variables usually configure before training a model.
# Learning Rate
# Number of Epochs
# Hidden Layers
# Hidden Units
# Activations Functions
#  

# # Hyperparameters Optimisation Techniques
# The process of finding most optimal hyperparameters in machine learning is called hyperparameter optimisation.
# Common algorithms include:
#
# Grid Search
#
# Random Search
#
# Bayesian Optimisation
#

# # Difference between SciKit Learn , Pytorch and Keras:
#
# SKlearn is a general machine learning library, built on top of NumPy. It features a lot of machine learning algorithms such as support vector machines, random forests, as well as a lot of utilities for general pre- and postprocessing of data. It is not a neural network framework.
#
# PyTorch is a deep learning framework, consisting of
#
# A vectorized math library similar to NumPy, but with GPU support and a lot of neural network related operations (such as softmax or various kinds of activations)
# Autograd - an algorithm which can automatically calculate gradients of your functions, defined in terms of the basic operations
# Gradient-based optimization routines for large scale optimization, dedicated to neural network optimization
# Neural-network related utility functions
# Keras is a higher-level deep learning framework, which abstracts many details away, making code simpler and more concise than in PyTorch or TensorFlow, at the cost of limited hackability. It abstracts away the computation backend, which can be TensorFlow, Theano or CNTK. It does not support a PyTorch backend, but that's not something unfathomable - you can consider it a simplified and streamlined subset of the above.
#
# In short, if you are going with "classic", non-neural algorithms, neither PyTorch nor Keras will be useful for you. If you're doing deep learning, scikit-learn may still be useful for its utility part; aside from it you will need the actual deep learning framework, where you can choose between Keras and PyTorch but you're unlikely to use both at the same time. This is very subjective, but in my view, if you're working on a novel algorithm, you're more likely to go with PyTorch (or TensorFlow or some other lower-level framework) for flexibility. If you're adapting a known and tested algorithm to a new problem setting, you may want to go with Keras for its greater simplicity and lower entry level.

# # Example for How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras

# +
# Important liberary if wanna import keras pipeline to sklearn model

import numpy as np

from scipy.ndimage import convolve

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn import  datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
import matplotlib.pyplot as plt

import os

# -

# # What Is the Difference Between Batch and Epoch?
# The batch size is a number of samples processed before the model is updated.
#
# The number of epochs is the number of complete passes through the training dataset.
#
# The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
#
# The number of epochs can be set to an integer value between one and infinity. You can run the algorithm for as long as you like and even stop it using other criteria besides a fixed number of epochs, such as a change (or lack of change) in model error over time.
#
# They are both integer values and they are both hyperparameters for the learning algorithm, e.g. parameters for the learning process, not internal model parameters found by the learning process.
#
# You must specify the batch size and number of epochs for a learning algorithm.
#
# There are no magic rules for how to configure these parameters. You must try different values and see what works best for your problem.
#
#

# # How to Tune the Training Optimization Algorithm
# Keras offers a suite of different state-of-the-art optimization algorithms.
#
# In this example, we tune the optimization algorithm used to train the network, each with default parameters.
#
# This is an odd example, because often you will choose one approach a priori and instead focus on tuning its parameters on your problem (e.g. see the next example).
#

# # How to Tune Learning Rate and Momentum
# It is common to pre-select an optimization algorithm to train your network and tune its parameters.
#
# By far the most common optimization algorithm is plain old Stochastic Gradient Descent (SGD) because it is so well understood. In this example, we will look at optimizing the SGD learning rate and momentum parameters.
#
# Learning rate controls how much to update the weight at the end of each batch and the momentum controls how much to let the previous update influence the current weight update.
#
# We will try a suite of small standard learning rates and a momentum values from 0.2 to 0.8 in steps of 0.2, as well as 0.9 (because it can be a popular value in practice).
#
# Generally, it is a good idea to also include the number of epochs in an optimization like this as there is a dependency between the amount of learning per batch (learning rate), the number of updates per epoch (batch size) and the number of epochs.

# # How to Tune Network Weight Initialization
# Neural network weight initialization used to be simple: use small random values.
#
# Now there is a suite of different techniques to choose from. Keras provides a laundry list.
#
# In this example, we will look at tuning the selection of network weight initialization by evaluating all of the available techniques.
#
# We will use the same weight initialization method on each layer. Ideally, it may be better to use different weight initialization schemes according to the activation function used on each layer. In the example below we use rectifier for the hidden layer. We use sigmoid for the output layer because the predictions are binary.

# # How to Tune the Neuron Activation Function
# The activation function controls the non-linearity of individual neurons and when to fire.
#
# Generally, the rectifier activation function is the most popular, but it used to be the sigmoid and the tanh functions and these functions may still be more suitable for different problems.
#
# In this example, we will evaluate the suite of different activation functions available in Keras. We will only use these functions in the hidden layer, as we require a sigmoid activation function in the output for the binary classification problem.
#
# Generally, it is a good idea to prepare data to the range of the different transfer functions, which we will not do in this case.

# # How to Tune Dropout Regularization
# In this example, we will look at tuning the dropout rate for regularization in an effort to limit overfitting and improve the model’s ability to generalize.
#
# To get good results, dropout is best combined with a weight constraint such as the max norm constraint.
#
# For more on using dropout in deep learning models with Keras see the post:
#
# Dropout Regularization in Deep Learning Models With Keras
# This involves fitting both the dropout percentage and the weight constraint. We will try dropout percentages between 0.0 and 0.9 (1.0 does not make sense) and maxnorm weight constraint values between 0 and 5.

# # How to Tune the Number of Neurons in the Hidden Layer
# The number of neurons in a layer is an important parameter to tune. Generally the number of neurons in a layer controls the representational capacity of the network, at least at that point in the topology.
#
# Also, generally, a large enough single layer network can approximate any other neural network, at least in theory.
#
# In this example, we will look at tuning the number of neurons in a single hidden layer. We will try values from 1 to 30 in steps of 5.
#
# A larger network requires more training and at least the batch size and number of epochs should ideally be optimized with the number of neurons.

# # Tips for Hyperparameter Optimization
# This section lists some handy tips to consider when tuning hyperparameters of your neural network.
#
# k-fold Cross Validation. You can see that the results from the examples in this post show some variance. A default cross-validation of 3 was used, but perhaps k=5 or k=10 would be more stable. Carefully choose your cross validation configuration to ensure your results are stable.
#
# Review the Whole Grid. Do not just focus on the best result, review the whole grid of results and look for trends to support configuration decisions.
#
# Parallelize. Use all your cores if you can, neural networks are slow to train and we often want to try a lot of different parameters. Consider spinning up a lot of AWS instances.
#
# Use a Sample of Your Dataset. Because networks are slow to train, try training them on a smaller sample of your training dataset, just to get an idea of general directions of parameters rather than optimal configurations.
# Start with Coarse Grids. Start with coarse-grained grids and zoom into finer grained grids once you can narrow the scope.
#
# Do not Transfer Results. Results are generally problem specific. Try to avoid favorite configurations on each new problem that you see. It is unlikely that optimal results you discover on one problem will transfer to your next project. Instead look for broader trends like number of layers or relationships between parameters.
#
# Reproducibility is a Problem. Although we set the seed for the random number generator in NumPy, the results are not 100% reproducible. There is more to reproducibility when grid searching wrapped Keras models than is presented in this post.

# +
# Function to create model, we will use it for Kerasclassifier
def create_keras_model():
    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu', input_dim = 60))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #compile
    model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    return model

# fix the no. of seeds for reproducability
seed = 7
np.random.seed(seed)

# load the dataset
print("the current directory we are working in {}" .format(os.getcwd()))
df = pd.read_csv('sonar_all_dataCopy1.csv', index_col= False)
print (type(df))

# split into train and test for X, y
X = df.iloc[:, 0:60]
y = df.iloc[:, 60:61]
print (X.shape, y.shape)
# create model 
model = KerasClassifier(build_fn=create_keras_model, epochs=150, batch_size=50, verbose=0)

# Define the grid search parameters

batch_size = [10, 30, 40 , 50, 80, 100]
epochs = [20, 30, 100]
#neurons = [1, 5, 10, 15, 20, 25, 30]
weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#activation = ['softmax', 'relu', 'sigmoid', 'linear']
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator= model , param_grid= param_grid, n_jobs=-1, cv = 5)
grid_result = grid.fit(X, y)
# -

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


