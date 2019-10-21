# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"nbpresent": {"id": "8b0d1366-6552-473d-9e29-2a9b1f283b8e"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Parallelising Neural Network Training with Keras

# + {"nbpresent": {"id": "ae7b2481-86af-4893-9753-02475b283430"}, "cell_type": "markdown"}
# <img src="./images/keras-logo.png" />

# + {"nbpresent": {"id": "ace66a88-c671-483a-b49c-346b26ac7601"}, "cell_type": "markdown"}
# **Note**: This notebook is adapted from chapter 13 in *Python Machine Learning* book, using Keras instead of TensorFlow. This notebook contains also (many) elements of book [*Deep Learning with Python*](https://www.manning.com/books/deep-learning-with-python), chapter 3 and 4. The Keras code is stored [here](https://github.com/fchollet/deep-learning-with-python-notebooks).

# + {"nbpresent": {"id": "91436083-c317-4bd4-ac12-e280a2298e95"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Keras

# + {"nbpresent": {"id": "bd4a1fe5-bba1-4c90-b406-d17801ea6257"}, "cell_type": "markdown"}
# - [Keras homepage](https://keras.io/)
# - Run Keras
#     * locally: 
#         - it is likely that your laptop doesn't have a graphical processing unit (GPU) suitable for use deep learning computations. This will result in longer training times for your model, but may be the only feasable option if you need to work offline.
#         - installing all necessary components may pose problems
#     * in the cloud using [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb): 
#         - this requires internet connection and a Google account. 
#         - You will have access to a GPU or TPU (if the software allows for use of TPU) 

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - TensorFlow
# -

# [TensorFlow](https://www.tensorflow.org/) (by Google)
# * Still one of the most used deep learning library (October 2019)
# * Version 2.0 out a short while ago (10 days ago) with Keras integrated
# * Version 1.X known for its poor documentation
# * Version 1.X tricky to use as reported frequently
# * Keras runs on top of Tensorflow (one out of currently three options)
# * In this year's edition of DAT300 we will use Keras that runs on top of TensorFlow 1.X

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - Theano
# -

# [Theano](http://deeplearning.net/software/theano/) (by [Mila](https://mila.quebec/en/) - [University of Montreal](https://www.umontreal.ca/en/))
# - said to be the fastest deep learning library in Python
# - development ceased in November 15, 2017
# - Keras runs on top of Theano (one out of currently three options)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - CNTK (Microsoft Cognitive Toolkit)
# -

# [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) (by Microsoft)
# - Keras runs on top of CNTK (one out of currently three options)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - PyTorch
# -

# [PyTorch](https://pytorch.org/)
# - based on [Torch](http://torch.ch/) (by [Facebook Artificial Intelligence Research Group](https://research.fb.com/category/facebook-ai-research/))
# - is said to be more "pythonic" than TensorFlow 1.X
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - Caffe2
# -

# [Caffe2](https://caffe2.ai/) by [University of California, Berkeley](https://bair.berkeley.edu/) and [Facebook Artificial Intelligence Research Group](https://research.fb.com/category/facebook-ai-research/)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Other deep learning libraries - Overview on Wiki
# -

# Comparison of various deep learning libraries at [wiki](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software).

# + {"nbpresent": {"id": "5d255640-4e48-48fe-9a93-788a8145ad18"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Overview

# + {"nbpresent": {"id": "6b1c5938-5349-4c06-9d37-56c080176d4e"}, "cell_type": "markdown"}
# - [For the unpatient - a quick look at a neural network built with Keras](#For-the-unpatient---a-quick-look-at-a-neural-network-built-with-Keras) - contains <font color=green>**COLAB NOTEBOOK**</font>
# - [Data representations for neural networks](#Data-representations-for-neural-networks) - contains <font color=green>**COLAB NOTEBOOK**</font>
#     - [Vector data](#Vector-data)
#     - [Time series or sequence data](#Time-series-or-sequence-data)
#     - [Image data](#Image-data)
#     - [Video data](#Video-data)
# - [Introduction to Keras](#Introduction-to-Keras)
#     - [Binary classification - Classifying movie reviews](#Binary-classification---Classifying-movie-reviews) - contains <font color=green>**COLAB NOTEBOOK**</font>
#     - [Multiclass classification - Classifying newswires](#Multiclass-classification---Classifying-newswires) - contains <font color=green>**COLAB NOTEBOOK**</font>
#     - [Regression - Predciting house prices](#Regression---Predciting-house-prices) - contains <font color=green>**COLAB NOTEBOOK**</font>
# - [Overfitting and underfitting: regularisation methods for ANN](#Overfitting-and-underfitting:-regularisation-methods-for-ANN) - contains <font color=green>**COLAB NOTEBOOK**</font>
# - [Choosing activation functions for multilayer networks](#Choosing-activation-functions-for-multilayer-networks)
# - [Summary](#Summary)

# + {"nbpresent": {"id": "973a0b6c-c02a-47cf-93d6-214574b216aa"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## For the unpatient - a quick look at a neural network built with Keras

# + {"nbpresent": {"id": "ab731075-b032-4850-8674-3c23c9da71ee"}, "cell_type": "markdown"}
# * Build our first neural network using the Keras library
# * More details on Keras later in this notebook
# * <font color=green>**COLAB NOTEBOOK**</font>: Data used in this [first example](https://colab.research.google.com/drive/15uWcS23hf-AhnuitFpjPo_lHWL3QehJZ): MNIST data (full set) of handwritten digits

# + {"nbpresent": {"id": "7c91e9a7-bb8e-4e36-a43b-9e6010a5bbb1"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Data representations for neural networks

# + {"nbpresent": {"id": "f8eee3fe-ea93-40ec-87cb-9eb5c9366818"}, "cell_type": "markdown"}
# * Previous example used data stored in multidimensional Numpy arrays, also called **tensors**.
# * All machine learning systems use tensors as their basic data structure
# * Tensors are fundamental to the field of machine learning - TensorFlow was named after them
#

# + {"nbpresent": {"id": "a1740ba8-51b2-42bd-a359-cda3e52e5b2c"}, "slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# * What are tensors?
#     - tensors are *containers for data* (almost always numerical data)
#     - tensors are a *generalisation of matrices* to an *arbitrary* number of dimensions
#     - a *dimension* is often called an axis in the context of tensors

# + {"nbpresent": {"id": "cad9e55a-f576-49d1-88a3-197af6f2cf64"}, "cell_type": "markdown"}
# <font color=green>**COLAB NOTEBOOK**</font>: code on [use of tensors](https://colab.research.google.com/drive/1gzCYp-W7lkWhrRVFf8KQjEByA6lsjHBR)

# + {"nbpresent": {"id": "50663c88-366d-42f7-9682-416c89ecc6ab"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Vector data

# + {"nbpresent": {"id": "c3c7292b-72ea-4819-85f0-c4ad9868a127"}, "cell_type": "markdown"}
# * This is the most common case
# * In such a dataset, each single data point can be encoded as a vector.
# * Thus a batch of data will be encoded as a 2D tensor (array of vectors)
# * 2D tensor: first axis is the *samples axis* and the second axis is the *features axis*.

# + {"nbpresent": {"id": "3c3b3a07-b081-4242-aeff-719ac5f10779"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Vector data -  Example 1

# + {"nbpresent": {"id": "8435301b-e916-4f7d-9f06-db9e144a15db"}, "cell_type": "markdown"}
# Actuarial dataset of people considering:
# * persons age
# * ZIP code
# * income
#
# Each person can be characterised as a vector of 3 values. A dataset of 100 000 people could be stored in a 2D tensor of shape `(100000, 3)`.

# + {"nbpresent": {"id": "98a413df-6017-426c-8195-e42bc6960066"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Vector data -  Example 2

# + {"nbpresent": {"id": "e9c4367b-ab75-4e1d-917c-eeb97d1ce379"}, "cell_type": "markdown"}
# A dataset of text documents, where each document is represented by counts of how many times each word appears in it (out of a dictionary of 20 000 common words).
#
# * Each document can be considered as a vector of 20 000 values
#
# If dataset consists of 500 documents, data could be stored in a 2D tensor of shape `(500, 20000)`

# + {"nbpresent": {"id": "d6aa12d3-f86f-4aed-85ba-b45b271e0ff9"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Time series or sequence data

# + {"nbpresent": {"id": "56cf3bdb-8807-4a69-b91a-9bd3a4e28aee"}, "cell_type": "markdown"}
# * Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a 3D tensor with an **explicit time axis**
# * Each sample can be encoded as a sequence of vectors (a 2D tensor)
# * Thus a batch of data will be encoded as a 3D tensor

# + {"nbpresent": {"id": "9a5c8ddf-f99e-4812-b84b-908dfa13158d"}, "cell_type": "markdown"}
# <img src="./images/fig_2.3.png" width="350"/>

# + {"nbpresent": {"id": "c79c36f6-f25e-416b-a4c2-9959b450ec76"}, "cell_type": "markdown"}
# [1] *F. Chollet*, Deep Learning with Python, chapter 2.2.10, Fig. 2.3. **A 3D timeseries data tensor**

# + {"nbpresent": {"id": "9f0d4d76-1b38-41dc-b649-ea678e106103"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Timeseries data -  Example

# + {"nbpresent": {"id": "167c6d03-7836-4ca0-adc7-3b63b0acadc9"}, "cell_type": "markdown"}
# A dataset of stockprices:
#
# * record current price of stock
# * record highest price in past minute
# * record lowest price in past minute
#
# Given a total of 390 minutes in a trading day, the data of one trading day will be stored in a 2D tensor of shape `(390 x 3)`. If there are 250 days of trading in one year then the data may be stored in a 3D tensor of shape `(250, 390, 3)`. In this case each sample would be the data of one trading day. 

# + {"nbpresent": {"id": "1b74754a-26c7-42b3-b03e-80c8870660bc"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Sequence data -  Example

# + {"nbpresent": {"id": "2831dfbb-a501-4567-a642-6cd3df6a7915"}, "cell_type": "markdown"}
# A dataset of tweets:
#
# * encode each tweet as a sequence of 280 characters out of an alphabet of 128 unique characters
# * each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character)
#
# Each tweet can be encoded as a 2D tensor of shape `(280, 128)`. Hence, a dataset of 1 million tweets can be stored in a 3D tensor of shape `(1000000, 280, 128)`.

# + {"nbpresent": {"id": "33207130-bf00-49d1-bbc6-19decbf10acf"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Image data

# + {"nbpresent": {"id": "e8ba5365-53d6-43d6-b69a-25c04ef96f59"}, "cell_type": "markdown"}
# * Images typically have three dimensions: height, width and colour depth
# * Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in 2D tensors, by convention image tensors are always 3D, with a onedimensional color channel for grayscale images
#
# A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape `(128, 256, 256, 1)`. A
# batch of 128 color images could be stored in a tensor of shape `(128, 256, 256, 3)`.

# + {"nbpresent": {"id": "686a5e30-6e1a-426e-8846-d8d613c6e283"}, "slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# <img src="./images/fig_2.4.png" width="400"/>

# + {"nbpresent": {"id": "8d561117-d8c0-424f-acbd-22971b80bd6c"}, "cell_type": "markdown"}
# [1] *F. Chollet*, Deep Learning with Python, chapter 2.2.11, Fig. 2.4. **A 4D image data tensor (channel-first convention)**

# + {"nbpresent": {"id": "306581a5-3681-422d-b927-114f977c0b0e"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Two conventions** for shapes of image tensors:
# * *channel-last* convention as used by TensorFlow `(samples, height, width, color_depth)`
# * *channel-first* convention as used by Theano `(samples, color_depth, height, width)`
# * Either, TensorFlow or Theano, may be used as the engine for Keras

# + {"nbpresent": {"id": "32922d54-f09b-4685-9338-49c4ddc6a5d7"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Video data

# + {"nbpresent": {"id": "5b64dd29-f187-46a8-829d-4b03149ca893"}, "cell_type": "markdown"}
# * Video data is one of the few types of real-world data for which you’ll need 5D tensors
# * A video can be understood as a sequence of frames, each frame being a color image
# * Because each frame can be stored in a 3D tensor `(height, width, color_depth)`, a sequence of frames can be stored in a 4D tensor `(frames, height, width, color_depth)`
# * Thus a batch of **different videos** can be stored in a 5D tensor of shape `(samples, frames, height, width, color_depth)`

# + {"nbpresent": {"id": "ffdef7ad-ecbd-48e0-943a-4db221e2e056"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Video data - Example

# + {"nbpresent": {"id": "e287bf1b-68af-4a6e-939f-8b5665d35eb1"}, "cell_type": "markdown"}
# Imagine you have a video with the following properties:
#
# * 60 second video
# * 144 x 256 (height, width)
# * 4 frames per second
#
# A batch of **four** such video clips would be stored in a tensor of shape `(4, 240, 144, 256, 3)`. That’s a total of 106 168 320 values! If the `dtype` of the tensor was `float32`, then each value would be stored in 32 bits, so the tensor would represent 405 MB, which is quite a lot. Videos in real life are much lighter, because they aren’t stored in `float32`, and they’re typically compressed by a large factor (such as in the MPEG format).

# + {"nbpresent": {"id": "1970300c-a616-41de-ba29-ca3bb2b83b82"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Introduction to Keras

# + {"nbpresent": {"id": "ac168c6c-bf66-4d47-ae8a-ad02145d64b9"}, "cell_type": "markdown"}
# This section covers:
# * Core components of neural networks
# * An introduction to Keras
# * Using neural networks to solve basic classification and regression problems
#
#
# * Three introductory examples of how to use neural networks to address real problems
#     - Classifying movie reviews as **positive** or **negative** (binary classification)
#     - Classifying news wires by **topic** (multiclass classification)
#     - Estimating the **price of a house**, given real-estate data (regression)

# + {"nbpresent": {"id": "2fcb1632-c60f-46ad-8b54-172992f4ae0c"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Anatomy of a neural network

# + {"nbpresent": {"id": "f808aaaa-cb10-4f61-9b98-2a3eb179dddd"}, "cell_type": "markdown"}
#  training a neural network revolves around the following objects:
#
# 1. **Layers**, which are combined into a network (or model)
# 2. The **input data** and **corresponding targets**
# 3. The **loss function**, which defines the feedback signal used for learning
# 4. The **optimizer**, which determines how learning proceeds

# + {"nbpresent": {"id": "022709f7-59a6-4e48-a210-43c28a4c8ce9"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/ANN.png" width="500"/>

# + {"nbpresent": {"id": "d23127c7-063e-4732-b1f0-ae0d6c022393"}, "cell_type": "markdown"}
# [1] *F. Chollet*, Deep Learning with Python, chapter 3.1, Fig. 3.1. **Relationship between the network, layers, loss function and optimiser.**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Layers: the building blocks of deep learning
# -

# * Layers are a fundamental data structure in neural networks
# * A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors
# * The layer’s **weights**, one or several tensors learned with stochastic gradient descent, contain the network’s **knowledge**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ##### Types of layers

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# Different layers are appropriate for different tensor formats and different types of data
# processing. For instance:
#
# * *simple vector data*, stored in 2D tensors of shape `(samples, features)`, is often processed by **densely** connected layers
#     - also called **fully** connected or **dense layers**
#     - represented by `Dense` class in Keras
# * *sequence data*, stored in 3D tensors of shape `(samples, timesteps, features)`, is typically processed by **recurrent layers** such as an `LSTM` layer
# * *Image data*, stored in 4D tensors, is usually processed by 2D **convolution layers** (`Conv2D`).

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# * Layers are the LEGO bricks of deep learning, a metaphor that is made explicit by frameworks like Keras
# * Building deep-learning models in Keras is done by clipping together **compatible layers** to form useful **data-transformation** pipelines.
# * The notion of **layer compatibility** here refers specifically to the fact that every layer will only accept **input tensors** of a **certain shape** and will return **output tensors** of a certain shape.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Code implementation**
# -

# <img src="./images/code_01.png" width="600"/>

# * Create a layer that accepts only 2D tensors (remember, that's what `Dense` layers do)
# * The `Dense` layer has `32` output or activation units
# * The first dimension of the 2D tensor is `784` (axis 0, the **batch dimension**, is unspecified, and thus any value would be accepted)
# * This layer will return a tensor where the first dimension has been transformed to be `32`

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Code implementation**
# -

# * This first layer can only be connected to a **downstream layer** that expects *32-dimensional* vectors as its input
# * With Keras, there is no reason to worry about compatibility, because the layers added to the models are **dynamically built** to **match the shape** of the **incoming layer**
# * The second layer didn’t receive an input shape argument — instead, it automatically inferred its input shape as being the output shape of the layer that came before

# <img src="./images/code_02.png" width="600"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Models: networks of layers
# -

# * A deep-learning model is a directed, acyclic graph of layers
# * The most common instance is a linear stack of layers, mapping a single input to a single output
# *  There is a broader variety of network topologies. Some common ones include the following:
#     - Two-branch networks
#     - Multihead networks
#     - Inception blocks

# * The **topology** of a network defines a **hypothesis space**
# * By choosing a network topology, you **constrain** your space of possibilities (hypothesis space) to a specific **series of tensor operations**, mapping input data to output data
# *  Picking the right network architecture is more an **art** than a **science**
# *  Although there are some best practices and principles you can rely on, **only practice** can help you become a proper neural-network architect

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Loss functions and optimizers: keys to configuring the learning process
# -

# Once the network architecture is defined, there are two more things that need to be defined.
#
# * **Loss function** (**objective function**): the quantity that will be minimized during training. It represents a measure of success for the task at hand
# * **Optimizer**: determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD)

# Choosing the right objective function for the right problem is extremely important: the network will take any shortcut it can, to minimize the loss. If the objective doesn’t fully correlate with success for the task at hand, your network will end up
# doing things you may not have wanted. 

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# For common problems such as classification, regression and sequence prediction, there are simple guidelines one can follow to choose the correct loss. For instance:
#
# * **binary crossentropy** for a **two-class classification** problem
# * **categorical crossentropy** for a **many-class classification** problem
# * **mean squared error** for a **regression** problem
# * **connectionist temporal classification (CTC)** for a sequence-learning problem
# * etc.
# -

# Only when working on truly new research problems one will have to develop own objective functions. In the next few chapters, we’ll detail explicitly which loss functions to choose for a wide range of common tasks. 

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Key features of Keras
# -

# Keras is a deep-learning framework for Python that provides a convenient way to define and train almost any kind of deep-learning model. It has the following key features:
#
# * It allows the same code to run seamlessly on CPU or GPU.
# * It has a user-friendly API that makes it easy to quickly prototype deep-learning models.
# * It has built-in support for convolutional networks (for computer vision), recurrent networks (for sequence processing), and any combination of both.
# * It supports arbitrary network architectures: multi-input or multi-output models, layer sharing, model sharing, and so on. This means Keras is appropriate for building essentially any deep-learning model, from a generative adversarial network to a neural Turing machine.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Keras, TensorFlow, Theano, and CNTK
# -

# Several different backend engines can be plugged seamlessly into Keras. Currently, the three existing backend implementations
# are the TensorFlow backend, the Theano backend, and the Microsoft Cognitive Toolkit (CNTK) backend. In the future, it’s likely that Keras will be extended to work with even more deep-learning execution engines

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/fig_3.3.png" width="600"/>
# -

# [1] *F. Chollet*, Deep Learning with Python, chapter 3.2.1, Fig. 3.3. **The deep learning software and hardware stack**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **About the software and hardware stack**
#
# * TensorFlow, CNTK, and Theano are some of the primary platforms for deep learning today
# * Any piece of code that you write with Keras can be run with any of these backends without having to change anything in the code: you can seamlessly switch between the two during development
# * It is recommended to use the TensorFlow backend as the default for most of deep-learning needs, because it’s the most widely adopted, scalable, and production ready
# * Via TensorFlow (or Theano, or CNTK), Keras is able to run seamlessly on both CPUs and GPUs
# * When running on CPU, TensorFlow is itself wrapping a low-level library for tensor operations called Eigen (http://eigen.tuxfamily.org)
# * On GPU, TensorFlow wraps a library of well-optimized deep-learning operations called the NVIDIA CUDA Deep Neural Network library (cuDNN
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Keras workflow
# -

# In the MNIST example we have already built our first Keras model. The general workflow is constructed like this: 
#
# 1. Define your training data: input tensors and target tensors.
# 2. Define a network of layers (or model ) that maps your inputs to your targets.
# 3. Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
# 4. Iterate on your training data by calling the `fit()` method of your model

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Defining a Keras model**
# -

# There are two ways to define a Keras model:
#
# 1. using the `Sequential` class (only for linear stacks of layers, which is the most common network architecture by far)
# 2. using the *functional* API (for directed acyclic graphs of layers, which lets you build completely arbitrary architectures)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Here’s a two-layer model defined using the `Sequential` class
# -

# <img src="./images/code_03.png" width="800"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# And here’s the same model defined using the *functional* API.
# -

# <img src="./images/code_04.png" width="800"/>

# With the *functional* API, one can manipulate the data tensors that the model processes and applying layers to this tensor as if they were functions.

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# Once your model architecture is defined, it doesn’t matter whether you used a `Sequential` model or the functional API. All of the following steps are the same. The learning process is configured in the compilation step, where you specify:
#
# * the optimiser for the model
# * the loss function for the model
# * the metrics to be monitored

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Here’s an example with a single loss function, which is by far the most common case:
# -

# <img src="./images/code_05.png" width="800"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Finally, the learning process consists of passing Numpy arrays of input data (and the corresponding target data) to the model via the `fit()` method, similar to what you would do in Scikit-Learn and several other machine-learning libraries:
# -

# <img src="./images/code_06.png" width="800"/>

# Over the next few chapters, you’ll build a solid intuition about what type of network architectures work for different kinds of problems, how to pick the right learning configuration, and how to tweak a model until it gives the results you want to see. We’ll look at three basic examples in sections: 
#
# * a two-class classification example
# * a many-class classification example 
# * a regression example

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Binary classification - Classifying movie reviews
# -

# Two-class classification, or binary classification, may be the most widely applied kind of machine-learning problem. In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the reviews.

# <font color=green>**COLAB NOTEBOOK**</font>: [Classifying movie reviews](https://colab.research.google.com/drive/1RNuKu22xavwkTB8FIVrsPMqbQFZXm1vZ)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Multiclass classification - Classifying newswires
# -

# <font color=green>**COLAB NOTEBOOK**</font>: [Classifying newswires](https://colab.research.google.com/drive/1Lpo0RoaYmKsjaYCU8ICW2NLsLBNx3MQ6)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Regression - Predciting house prices
# -

# <font color=green>**COLAB NOTEBOOK**</font>: [Predicting house prices](https://colab.research.google.com/drive/1IN1DmVEnx_mHgexOzR2Zyu6WbzWKZT5N)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Overfitting and underfitting: regularisation methods for ANN
# -

# <font color=green>**COLAB NOTEBOOK**</font>: [Regularisation in Keras](https://colab.research.google.com/drive/1Da_aRUmjdPUTOgWXfnVIcr5l9HAiIIWz)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Choosing activation functions for multilayer networks
# -

# * Technically, one can use **any function** as an activation function in multilayer neural networks as long as it is **differentiable**
# * One could can even use **linear** activation functions, such as in Adaline, but ...
#     - would **not** be very useful to use linear activation functions for both hidden and output layers 
#     - to tackle complex problems one needs to introduce **non-linearity**
#     - the **sum of linear functions** would yield only **another** linear function

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### An overview of various activation functions
# -

# <img src="./images/overview_actfunc.png" width="600"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# * The **logistic activation function** (which we often called *sigmoid function*) mimics the concept of a neuron in a brain most closely - think of it as the probability of whether a neuron fires or not
# * However, logistic activation functions can be problematic
#     - When net input $\textbf{z}$ is **highly negative**, $\phi{(\textbf{z})}$ would be close to zero
#     - If $\phi{(\textbf{z})}$ is close to zero the neural network would learn **very slowly**
#     - More slowly learning could lead to the neural network **getting trapped in local minima** during traning

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Logistic function recap
# -

# * The **logistic function** is a special case of a **sigmoid function**
# * We can use a logistic function to model the probability that sample $\textbf{x}$ belongs to the positive class (class 1) in a **binary classification** task
# * The net input $z$ is shown in the following equation:

# <img src="./images/netInputZ.png" width="800"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# The logistic function will compute the following:
# -

# <img src="./images/phiOfZ.png" width="400"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### Example using two-dimensional data

# +
# Import necessary modules
import numpy as np

# Generate some two-dimensional data including bias and weight vector w
X = np.array([1, 1.4, 2.5]) ## first value must be 1, since it represents bias
w = np.array([0.4, 0.3, 0.5])

# Define function for computation of net input z
def net_input(X, w):
    return np.dot(X, w)

# Define logistic function
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

# Define phi(z) activation function
def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

# Compute probability of x belonging to positive class (y = 1)
print('P(y=1|x) = %.3f' % logistic_activation(X, w))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# However, an output layer consisting of **multiple logistic activation units** does not produce meaningful, interpretable probability values. This is illustrated by code below:

# +
# NOTE: This code was adapted to correspond with equations discussed in chapter 12

# W : array with shape = (n_hidden_units + 1, n_output_units)
#     note that the first row are the bias units
W = np.array([[1.1, 0.2, 0.6],
              [1.2, 0.4, 1.5],
              [0.8, 1. , 1.2],
              [0.4, 0.2, 0.7]])

# A : data array with shape = (n_samples, n_hidden_units + 1)
#     note that the first column of this array must be 1
A = np.array([[1, 0.1, 0.4, 0.6]])

# Compute net input Z and probabilities 
Z = np.dot(A[0], W)
y_probas = logistic(Z)

print('Net Input: \n', Z)

print('Output Units:\n', y_probas)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# * The resulting values **cannot** be interpreted as **probabilities** for a three-class problem
# * The reason for this is that they **do not** sum up to 1
# * Usually, this is not of concern when we use the model to predict class membership
# * One to predict class membership is to assign sample to **maximum value** of $\textbf{Z}$

# + {"slideshow": {"slide_type": "fragment"}}
y_class = np.argmax(Z, axis=0)
print('Predicted class label: {0}'.format(y_class))


# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# * In certain contexts, it can be useful to compute meaningful class probabilities for multiclass predictions 
# * In the next section, we will take a look at a **generalization** of the logistic function, the ``softmax`` function, which can help us with this task

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Estimating class probabilities in multiclass classification via the ``softmax`` function
# -

# * In previous sections: obtain a class label using the ``argmax`` function
# * The ``softmax`` function is in fact a soft form of the ``argmax`` function; instead of giving a single class index, it provides the **probability of each class**
# * The ``softmax`` function allows for computation of **meaningful** class probabilities in **multiclass** settings (multinomial logistic regression)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# In ``softmax``, the probability of a particular sample with net input $z$ belonging to the $i$th class can be computed with a normalization term in the denominator, that is, the sum of all $M$ linear functions.
# -

# <img src="./images/softmax_eq.png" width="500"/>

# The ``softmax`` function coded in Python:

# +
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
# -

np.sum(y_probas)

np.argmax(y_probas)

# * The predicted class label is the same as when we applied the ``argmax`` function to the logistic output
# * Intuitively, it may help to think of the ``softmax`` function as a *normalized* output that is useful to obtain meaningful **classmembership predictions** in multiclass settings

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Broadening the output spectrum using a hyperbolic tangent
# -

# * Another *sigmoid function* that is often used in the **hidden layers** of artificial neural networks is the **hyperbolic tangent** (commonly known as ``tanh``)
# * ``tanh`` can be interpreted as a rescaled version of the logistic function

# <img src="./images/log_&_tanh.png" width="700"/>
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Advantage of the hyperbolic tangent over the logistic function**
#
# * It has a **broader output spectrum** and ranges in the **open interval** (-1, 1)
# * This can improve the convergence of the back propagation algorithm [Neural Networks for Pattern Recognition, C. M. Bishop, Oxford University Press, pages: 500-501, 1995](https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/neural_networks_pattern_recognition.pdf)
# * In contrast, the logistic function returns an output signal that ranges in the open interval (0, 1)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# For an intuitive comparison of the logistic function and the hyperbolic tangent, let's plot the two sigmoid functions:

# + {"slideshow": {"slide_type": "slide"}}
# %matplotlib inline
import matplotlib.pyplot as plt

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')

plt.plot(z, tanh_act,
         linewidth=3, linestyle='--',
         label='tanh')

plt.plot(z, log_act,
         linewidth=3,
         label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig('images/13_03.png')
plt.show()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# * The shapes of the two sigmoidal curves look very similar
# * however, the ``tanh`` function has 2× larger output space than the ``logistic`` function

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# * Note that we implemented the ``logistic`` and ``tanh`` functions verbosely for the purpose of illustration
# * In practice, we can use NumPy's ``tanh`` function to achieve the same results:

# + {"slideshow": {"slide_type": "fragment"}}
tanh_act = np.tanh(z)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# In addition, the logistic function is available in SciPy's special module:

# + {"slideshow": {"slide_type": "fragment"}}
from scipy.special import expit
log_act = expit(z)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Rectified linear unit activation
# -

# * ``tanh`` and ``logistic`` activations suffer from **vanishing gradient problem**
# * This means the **derivative of activations** with respect to net input **diminishes** as $z$ becomes large
# * As a result, **learning weights during the training phase** become **very slow** because the gradient terms may be **very close to zero**
# * ReLU activation addresses this issue

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# #### **Example**
#
# * Initial net input $z_{epoch1} = 20$
# * After weight update net input changes to $z_{epoch2} = 25$
# * Using ``tanh`` as activation function we get $\phi{(z_{epoch1})} \approx 1.0$ and $\phi{(z_{epoch2})} \approx 1.0$
#
# * This means the **derivative of activations** with respect to net input **diminishes** as $z$ becomes large
# * As a result, learning weights during the training phase become **very slow** because the **gradient terms** may be **very close to zero**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Mathematically, ReLU is defined as follows:
# -

# <img src="./images/relu.png" width="400"/>

# * ReLU is still a nonlinear function that is good for learning complex functions with neural networks
# * Besides this, the derivative of ReLU, with respect to its input, is always 1 for positive input values
# * Therefore, it **solves** the problem of vanishing gradients, making it **suitable for deep neural networks**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Loss function: Cross entropy

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# * Cross entropy loss, or log loss, measures the **performance of a classification model** whose output represents a **probability**, that is, a value between 0 and 1. 
# * Cross entropy loss **increases** as the predicted probability **diverges** from the actual label. So predicting a probability of for example .017 when the actual observation label is 1 would result in a **high** loss value. 
# * A perfect model would have a log loss of 0. 
#
#

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# **Choice of cross entropy in Keras**
#
# * Binary classification problems: use binary cross entropy (``binary_crossentropy`` in Keras)
# * Multi-class classification problems: use categorical cross entropy (``categorical_crossentropy`` in Keras)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Binary cross entropy
# -

# <img src="./images/binary_logloss.png" width="700"/>

# + {"slideshow": {"slide_type": "slide"}}
# Import modules
import numpy as np
import math as m
import matplotlib.pyplot as plt
# %matplotlib inline


# Define binary cross entropy
def binary_crossEntropy(yHat, y):
    term_1 = y * m.log(yHat)#; print(term_1)
    term_2 = (1 - y) * m.log(1 - yHat)#; print(term_2)
    
    return -(term_1 + term_2)

# Get values between 0 and 1 representing probabilities for which
# we will compute log loss for
values = np.linspace(0.001, .999, 999)

# Compute log losses
logloss = []
for val in values:
    logloss.append(binary_crossEntropy(val, 1))

# Plot log loss, given that true class label is 1
plt.plot(values, logloss, '.')
plt.title("log loss when true class label = 1")
plt.xlabel("predicted probablility")
plt.ylabel("log loss")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Categorical cross entropy
# -

# <img src="./images/categorical_logloss.png" width="600"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/cross_entropy.png" width="900"/>

# + {"slideshow": {"slide_type": "slide"}}
# Import modules
import numpy as np

# Define categorical cross entropy
def categorical_crossEntropy(yHat, y):
    return - np.sum(y * np.log(yHat))

# Define some toy data
y_pred = np.array([0.05, 0.9, 0.05])
y = np.array([0, 1, 0])

# Compute cross entropy
r2 = categorical_crossEntropy(y_pred, y)
print(r2)
# -

# <img src="./images/categorical_logloss.png" width="600"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Summary
# -

# Things you have learned in this series of lectures

# * Overview over different deep learning tools in using Python
# * Basic anatomy/architecture of ANN
# * How to build and train models for 
#     * binary and multiclass classification
#     * regression with few samples using KFold cross validation
# * Choose appropriate activation functions for specific problems
# * Basic preprocessing of text data
# * Regularisation methods for ANN
