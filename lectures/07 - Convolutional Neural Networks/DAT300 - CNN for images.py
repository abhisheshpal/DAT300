# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Chapter 15 - Classifying Images with Deep Convolutional Neural Networks

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# ### Overview

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - [Building blocks of convolutional neural networks](#Building-blocks-of-convolutional-neural-networks)
#   - [Understanding CNNs and learning feature hierarchies](#Understanding-CNNs-and-learning-feature-hierarchies)
#   - [Performing discrete convolutions](#Performing-discrete-convolutions)
#     - [Performing a discrete convolution in one dimension](#Performing-a-discrete-convolution-in-one-dimension)
#     - [The effect of zero-padding in convolution](#The-effect-of-zero-padding-in-convolution)
#     - [Determining the size of the convolution output](#Determining-the-size-of-the-convolution-output)
#     - [Performing a discrete convolution in 2D](#Performing-a-discrete-convolution-in-2D)
#     - [Sub-sampling](#Sub-sampling)
#   - [Putting everything together to build a CNN](#Putting-everything-together-to-build-a-CNN)
#     - [Implementing a CNN in Keras](#Implementing-a-deep-convolutional-neural-network-using-Keras)
#     - [Loading and preprocessing the data](#Loading-and-preprocessing-the-data)

# + {"slideshow": {"slide_type": "skip"}}
from IPython.display import Image
# %matplotlib inline

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Building blocks of convolutional neural networks 
# - Family of models inspired by the visual cortex of the human brain
#     - object recognition
# - Developed in the earliy 1990
# - Important building block in most DNNs for image classification

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Understanding CNNs and learning feature hierarchies
# - Extraction of relevant features (1D, 2D, ...)
#     - Translational invariant patterns

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/CNN_local_patterns.png" alt="Feature map" style="width: 300px;"/>
# MNIST letter borrowed from "Deep Learning with Python" (F. Chollet)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Combining features in a hierarchy
# - Low-level features: edges, simple patterns
# - High-level features: shapes, objects

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/CNN_cat.png" alt="Feature map" style="width: 700px;"/>
# Cat borrowed from "Deep Learning with Python" (F. Chollet)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Feature map
# - Each element is connected to a small patch of pixels
# - Weights are shared across all pathes
# - Size of filters and step sizes control overlap and size of features
# - Number of filters controls the number of features that can be learned
# <img src="./images/15_01.png" alt="Feature map" style="width: 700px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Parameters
# - Each convolutional filter has #parameters = #elements.
#   - Banks of filters and a set of input channels increases this.
# - Pooling layers: simplify/decrease dimensionality, e.g. with max: no parameters.
# - Fully connected layers (multilayer perceptron): #parameters from (IxJ) weight matrix.
# - Understanding the complexity will be revisited later.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Performing discrete convolutions  
# Notation:
# - Dimension: $\bf{A}_{\it{n_1 \times n_2}}$
# - Indexing: $\bf{A}[\it{i,j}\bf{]}$
# - Convolutional operator: $\ast$
#   - Not to be confused with dot product, matrix product, Hadamard, Kronecker, Khatri-Rao, ...

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ###  Performing a discrete convolution in one dimension
# - (Discrete) 1D convolution between input (signal) $x$ and filter $w$ is defined as:
# $$y=x \ast w \rightarrow y[i] = \sum^{+\infty}_{k=-\infty}x[i-k]w[k]$$
# - i.e. each element of the output is the product of a subset of the input and each filter coefficient
# - Infinities are in practice limited to padding with zeros, e.g. to the size of the filter

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# <img src="./images/15_02.png" alt="1D convolution" style="width: 700px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Convolution with padding:
# - Assume input $x$ of size $n$ and filter $w$ of size $m$.
# - A padded vector $x^p$ with size $n+2p$ results in the more practical formula:
# $$y=x \ast w \rightarrow y[i] = \sum^{k=m-1}_{k=0}x^p[i+m-k]w[k]$$
# - Even more practical: Flip the filter to get a dot product notation which can be repeated like a sliding window:
# $$y[i] = x[i:i+m].w^r$$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Example
# - $n=8, m=4, p=0, stride=2$
# <img src="./images/15_03.png" alt="1D convolution steps" style="width: 900px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### The effect of zero-padding in convolution
# - Full padding: $p=m-1$, gives equal usage of elements, increases output size.
# - Same padding: $p$ and stride combined to achieve equal input and output size (most used in DNN).
# - Valid padding: $p=0$, shrinks output and has unequal usage of elements (many layers shrinks too much).
# - First two are recommended, possibly together with pooling for downsampling.

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/15_11.png" alt="Padding" style="width: 700px;"/> 

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Determining the size of the convolution output
# $$o = \left \lfloor{\frac{n+2p-m}{s}}\right \rfloor +1 $$  
#   
# Example:  
# $$n=10,m=5,p=2,s=1$$  
#   
# $$o = \left \lfloor{\frac{10+2\times2-5}{1}}\right \rfloor +1 = 10$$  
# #Input == #Output => "same"

# + {"slideshow": {"slide_type": "slide"}}
# Naïve 1D convolution:
import numpy as np

def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
    return np.array(res)

## Testing:
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
print('Conv1d Implementation: ', 
      conv1d(x, w, p=2, s=1))
print('Numpy Results:         ', 
      np.convolve(x, w, mode='same'))

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# Mental arithmetic (aided by paper/computer if needed): Compute the 1st and 4th output manually with your neighbour.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### The effects of 1D convolutional filters

# + {"slideshow": {"slide_type": "-"}}
import matplotlib.pyplot as plt
import numpy as np
x = [-2, -2, -2, -2, -2, -1, 1, 3, 4, 3, 1, -1, -2, -2, -2, -2, -2]

plt.plot(x, label='x', lw=4); plt.plot(np.zeros(np.shape(x)), c='#000000')
#w1 = [-1, -1, -1]; plt.plot(np.convolve(x, w1, mode='same'), label='w1', ls='--')
#w2 = [1, 0.5, 1];    plt.plot(np.convolve(x, w2, mode='same'), label='w2', ls=':')
w3 = [-1, 0, 1];   plt.plot(np.convolve(x, w3, mode='same'), label='w3', ls='-.')
w4 = [1, 0, -1];   plt.plot(np.convolve(x, w4, mode='same'), label='w4')
plt.legend(); plt.ylim([-10.5, 10.5])
plt.show()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Performing a discrete convolution in 2D
# Just like 1D, but in 2D. :-)
# - $X_{n_1 \times n_2}$, $W_{m_1 \times m_2}$  
# $$Y=X \ast W \rightarrow Y[i,j] = \sum^{+\infty}_{k_1=-\infty}\sum^{+\infty}_{k_2=-\infty}X[i-k_1,j-k_2]W[k_1,k_2]$$  
#   
# - Zero-padding, filter rotation and strides still apply

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/15_04.png" alt="2D convolution" style="width: 700px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - $n_1=3,n_2=3,m_1=3,m_2=3,p=1,s=2$
# - Rotated filter: W_rot=W[::-1,::-1]
# <img src="./images/15_05.png" alt="2D convolution steps" style="width: 900px;"/>

# + {"slideshow": {"slide_type": "slide"}}
# Naïve 2D convolution:
import numpy as np
import scipy.signal


def conv2d(X, W, p=(0,0), s=(1,1)):
    W_rot = np.array(W)[::-1,::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0] + X_orig.shape[0], 
             p[1]:p[1] + X_orig.shape[1]] = X_orig

    res = []
    for i in range(0, int((X_padded.shape[0] - 
                           W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - 
                               W_rot.shape[1])/s[1])+1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0], j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))
    return(np.array(res))
    
X = [[1, 3, 2, 4], [5, 6, 1, 3], [1 , 2,0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
print('Conv2d Implementation: \n', 
      conv2d(X, W, p=(1,1), s=(1,1)))

print('Scipy Results:         \n', 
      scipy.signal.convolve2d(X, W, mode='same'))

# Much more efficient solutions are implemented, e.g. in Tensorflow, 
# especially quick for typical filter sizes of 1x1, 3x3, 5x5

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Convolution filter applied to letter 'o'
# <img src="./images/CNN_filter2D.png" alt="Sub-sampling" style="width: 700px;"/>
# Response map borrowed from "Deep Learning with Python" (F. Chollet)
# -

# ## A note on implementation
# - In practice filters are not flipped.
#   - Tensorflow, PyTorch, ...
# - Convolution -> Cross-correlation.
# - Quicker computations, no side-effects for deep learning purposes.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Convolution flow
# Input: $n_1=5,n_2=5,input depth=2$, Convolution $m_1=3,m_2=3, p=0, s=1, output depth=3$ (3 filters)
# <img src="./images/CNN_flow.png" alt="Sub-sampling" style="width: 700px;"/>
# Flow borrowed from "Deep Learning with Python" (F. Chollet)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Sub-sampling
# - Max-pooling and mean-pooling (average-pooling)
# - $P_{n_1 \times n_2}$, the size of the neighbourhood for pooling
# - Introduces robustness to small local changes due to noise or other minor variations.
# - Dimensional reduction for higher computational efficency and reduced overfitting.
# - Usually non-overlapping, i.e. stride-size = pooling-size, but deviations occur.

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/15_06.png" alt="Sub-sampling" style="width: 700px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Alternative subsampling by convolution
# $$o = \left \lfloor{\frac{n+2p-m}{s}}\right \rfloor +1 $$  
# - $n$=input width (e.g. 100)
# - $p$=padding
# - $m$=filter size
# - $s$=stride 
#   
# How could convolutions be used to output ~1/3 of the input size?

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Putting everything together to build a CNN 
# - Same building structure as previous NN
#     - Convolution as pre-activation ($A = W \ast X + b$ instead of $A = Wx+b$)
#     - Activation: $H = \phi(A)$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Working with multiple input or color channels
# - Several layers of $X$:
#     - RGB: Rank-3 tensor / three-dimensional array $X_{N_1 \times N_2 \times C_{in}}$
#     - Grayscale: $C_{in}=1$
# - Images can typically be read in as data type 'uint8' ($2^8=256$ integer values) to save space.

# + {"slideshow": {"slide_type": "slide"}}
# imageio.imread replaces
# scipy.misc.imread in scipy >= 1.2.0

import imageio
try:
    img = imageio.imread('./example-image.png', pilmode='RGB')
except AttributeError:
    s = ("imageio.imread requires Python's image library PIL"
         " You can satisfy this requirement by installing the"
         " userfriendly fork PILLOW via `pip install pillow`.")
    raise AttributeError(s)

# + {"slideshow": {"slide_type": "slide"}}
plt.imshow(img)
plt.show()

# + {"slideshow": {"slide_type": "-"}}
print('Image shape:', img.shape)
print('Number of channels:', img.shape[2])
print('Image data type:', img.dtype)

print(img[100:102, 100:102, :])

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Convolution in multiple channels
# - Separate kernel per layer
# - Convolution per layer, then sum over layers
# <img src="./images/CNN_multi1.png" alt="Convolution and pooling" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - Multiple feature maps ($W_{m_1 \times m_2 \times C_{in} \times C_{out}}$)
# <img src="./images/CNN_multi2.png" alt="Convolution and pooling" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/15_07.png" alt="Convolution and pooling" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Parameter-sharing and sparse-connectivity
# - CNN trainable parameters here: $m_1 \times m_2 \times 3 \times 5 + 5$ (last 5 for bias units)
#     - No trainable parameter for pooling
#     - Input of size $n_1 \times n_2 \times 3$, and assuming mode='same', gives feature map of size $n_1 \times n_2 \times 5$

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/15_07.png" alt="Convolution and pooling" style="width: 500px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Parameter-sharing and sparse-connectivity
# - CNN trainable parameters here: $m_1 \times m_2 \times 3 \times 5 + 5$ (last 5 for bias units)
# - Corresponding number of parameters with a fully connected layer: $(n_1 \times n_2 \times 3) \times (n_1 \times n_2 \times 5) = (n_1 \times n_2)^2 \times 3 \times 5$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Regularizing a neural network
# - The size of a well-performing network is a challenging problem.
# - 'Capacity': level of complexity that can be learned
#     - Too small => under fit, cannot learn underlying structure if complex
#     - Too large => over fit, perfect learning - bad prediction
# - One strategy: overfit sligthly, then regularize
#     - L2 regularization
#     - Norm constraint on weights (e.g. kernel_constraint=maxnorm(4))
#     - Add Gaussian noise to weights
#     - Droput

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Dropout
# - Randomly drop hidden units of higher levels, e.g with $p_{drop}=0.5$, when training
#     - Activations must be scaled to compensate for dropout
# - Forces the network to learn a redundant representation of the data
#     - More general and robust as units may be dropped at any time
# - Predict using all hidden units
# - Link to ensemble learning as each dropout set corresponds to a model and the final prediction is the average over these

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/15_08.png" alt="Dropout" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# Filter-wise dropout of convolutions are possible in some implementations.

# + {"slideshow": {"slide_type": "notes"}, "cell_type": "markdown"}
# ---> Ended here Monday 21.10

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Why image classification is difficult
# <img src="./images/Difficult1.jpg" alt="Difficult 1" style="width: 500px;"/>
# <img src="./images/Difficult2.jpg" alt="Difficult 2" style="width: 500px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Traditional pre-processing
# <img src="./images/Trad_preprocessing.jpg" alt="Traditional preprocessing" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Implementing a deep convolutional neural network using Keras
# - Example reusing the MNIST data, $n_1 \times n_2 \times c_{in} = 28 \times 28 \times 1$ (grayscale).
# - $5 \times 5$ kernels, 32 and 64 output feature maps, fully connected ($1024 \times 1024$), then fully connected ($1024 \times 10$) acting as softmax (no padding)

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <img src="./images/15_09.png" alt="Multilayer CNN" style="width: 1000px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Local modelling below.  
# [Colab notebook](https://drive.google.com/file/d/1VGzUW849GlF208pduiQ6UNBmNV4e8y9h/view?usp=sharing)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Loading and preprocessing the data

# + {"slideshow": {"slide_type": "-"}}
## unzips mnist

import sys
import gzip
import shutil
import os


if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./')
                if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read())

# + {"slideshow": {"slide_type": "slide"}}
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                                % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_data, y_data = load_mnist('./', kind='train')
print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
X_test, y_test = load_mnist('./', kind='t10k')
print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

X_train, y_train = X_data[:50000,:], y_data[:50000]
X_valid, y_valid = X_data[50000:,:], y_data[50000:]

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Standardize each collection with regard to training
# - Pixel-wise means
# - Set-wise standard deviations (avoiding 0 division for constant pixels)

# + {"slideshow": {"slide_type": "-"}}
# Standardize data based on training data
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_data, y_data, X_train, X_valid, X_test

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Encodig and reshaping to 4D tensors

# + {"slideshow": {"slide_type": "-"}}
from sklearn.preprocessing import OneHotEncoder #, LabelEncoder
#label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y_train)
one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
Y_train = one_hot_encoder.fit_transform(y_train.reshape(-1,1))
Y_valid = one_hot_encoder.transform(y_valid.reshape(-1,1))
Y_test  = one_hot_encoder.transform(y_test.reshape(-1,1))

X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], 28, 28, 1)) # Grayscale = 1
X_valid_centered = X_valid_centered.reshape((X_valid_centered.shape[0], 28, 28, 1))
X_test_centered  = X_test_centered.reshape((X_test_centered.shape[0], 28, 28, 1))
print(X_train_centered.shape)
print(Y_train.shape)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Set up sequential KNN model in Keras

# +
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=[5,5], padding='valid', activation='relu', input_shape=(X_train_centered.shape[1:])))
model.add(MaxPooling2D(pool_size=[2,2], padding='same'))
model.add(Conv2D(filters=64, kernel_size=[5,5], padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=[2,2], padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='softmax'))
model.summary()

# + {"slideshow": {"slide_type": "notes"}}
# Requires installation of graphViz
import pydot

# + {"slideshow": {"slide_type": "notes"}}
# #!pip install pydot
from keras.utils import plot_model
plot_model(model, to_file="mod_plot.png")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Compile model and prepare TensorBoard

# + {"slideshow": {"slide_type": "slide"}}
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpoint = ModelCheckpoint("models", save_best_only=True) #/MNIST.s

# + {"slideshow": {"slide_type": "fragment"}}
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=64, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
# To show TensorBoard, write e.g. tensorboard --logdir path_to_logdir --host 127.0.0.1 --port 80
# and navigate to 127.0.0.1 through a browser

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Training of the model
# - ~30 minutes on KHL's CPU
# -

# Load the TensorBoard notebook extension
# %load_ext tensorboard

import datetime
from tensorflow.keras.callbacks import TensorBoard
logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# + {"slideshow": {"slide_type": "-"}}
history = model.fit(X_train_centered, Y_train, batch_size=64, epochs=20, 
          verbose=1, shuffle=True, #          verbose=1, shuffle=True,
          validation_data=(X_valid_centered, Y_valid))#, callbacks=[checkpoint])#, tensorboard])
# A path bug in TensorFlow 2.0 makes callbacks in Windows difficult

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Plot history

# + {"slideshow": {"slide_type": "-"}}
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Evaluate on test data

# + {"slideshow": {"slide_type": "-"}}
# Explicit saving of the model for later use
model.save('models/MNIST.h5')

# + {"slideshow": {"slide_type": "-"}}
# Prediction
# from keras.models import load_model
# model = load_model('models/MNIST.h5')
model.evaluate(X_test_centered, Y_test)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Probabilities of each class

# + {"slideshow": {"slide_type": "-"}}
# Predict with probabilities
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(model.predict(X_test_centered[:4,:]))
plt.show()
model.predict(X_test_centered[:4,:])

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Confusion matrix

# + {"slideshow": {"slide_type": "-"}}
from sklearn.metrics import confusion_matrix
import numpy as np
confusion_matrix(y_test,np.argmax(model.predict(X_test_centered), axis=1), labels = list(range(10)))

# + {"slideshow": {"slide_type": "slide"}}
# Functions from https://github.com/philipperemy/keras-activations/blob/master/keract/keract.py and
# https://github.com/philipperemy/keras-activations/blob/master/examples/utils.py

import keras.backend as K
import matplotlib.pyplot as plt

def get_activations(model, model_inputs, layer_name=None):
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
    outputs = [output for output in outputs if 'input_' not in output.name]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    activations = [func(list_inputs)[0] for func in funcs]
    layer_names = [output.name for output in outputs]

    result = dict(zip(layer_names, activations))
    return result


def display_activations(activations):
    import numpy as np
    import matplotlib.pyplot as plt

    layer_names = list(activations.keys())
    activation_maps = list(activations.values())
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.title(layer_names[i])
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()

def print_names_and_shapes(activations):  # dict
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations.shape)
        print('')


def print_names_and_values(activations):  # dict
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations)
        print('')


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Layer names and shapes

# + {"slideshow": {"slide_type": "-"}}
print_names_and_shapes(get_activations(model, X_test_centered[:123,:]))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Activations of examples

# + {"slideshow": {"slide_type": "-"}}
sampleNum = 1
plt.figure()
plt.imshow(X_test_centered[sampleNum,:,:,0], interpolation='None', cmap='jet')
plt.show()
print("Ground truth: {}".format(y_test[sampleNum]))
plt.figure(figsize=(10, 10), dpi=400)
display_activations(get_activations(model, X_test_centered[sampleNum:sampleNum+1,:])) #,'conv2d_2'))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Weights from layers

# + {"slideshow": {"slide_type": "-"}}
layer = 0
filter = 9
weights, biases = model.layers[layer].get_weights()
print(weights.shape)
sns.heatmap(weights[:,:,0,filter])


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Visualizing convolution filters
# - Generate random patterns of maximum activation

# + {"slideshow": {"slide_type": "-"}}
# This function is made for RGB images, but works okay for grayscale too.
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model, layer_name, filter_index, size=28):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


# + {"slideshow": {"slide_type": "slide"}}
filter = 0
layer_name = 'conv2d_2'
pat = generate_pattern(model, layer_name, filter,50)
plt.imshow(np.squeeze(pat))
plt.show()

# + {"slideshow": {"slide_type": "slide"}}
filter = 0
layer_name = 'conv2d_1'
#pat = generate_pattern(model, layer_name, filter)
#print(pat.shape)
size = 50
margin = 3
results = np.zeros((4 * size + 3 * margin, 8 * size + 7 * margin, 1))
for i in range(4):
    for j in range(8):
        filter_img = generate_pattern(model, layer_name, i + (j * 4), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
            vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 10))
plt.imshow(np.squeeze(results))

# + {"slideshow": {"slide_type": "slide"}}
filter = 0
layer_name = 'conv2d_2'
#pat = generate_pattern(model, layer_name, filter)
#print(pat.shape)
size = 50
margin = 3
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
            vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(np.squeeze(results))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Pipelining
# - On-the-fly batch-wise pre-processing 
#     - Read from disk
#     - Convert to correct format
#     - Reshape
#     - Agument training set

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=False)
datagen.fit(X_train_centered)
train_generator = datagen.flow(np.array(X_train_centered), np.array(Y_train), 
                               batch_size=64)
# flow_from_directory for direct import from image files (can include resizing)
# flow_from_dataframe for direct import based on a list of file names (numbering counted alphabetically)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Applying the image generator

# + {"slideshow": {"slide_type": "-"}}
historyFlow = model.fit_generator(
    train_generator,
    epochs=10, steps_per_epoch=len(X_train_centered) / 64,
    validation_data=(np.array(X_valid_centered), np.array(Y_valid)), 
    validation_steps=len(X_valid_centered) / 64)


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Building on a pretrained network
# - Reuse existing networks for new/tuned purposes
#     - Called applications in Keras
# - Main tasks of neural networks on images:
#     - Generate meaningful features
#     - Combine into objects
#     - Distinguish between types of objects
# - Strategy:
#     - Strip final dense layer(s) (softmax)
#     - Freeze network parameters
#     - Train new dense layers for specific purpose

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Available networks in Keras
# - Xception
# - VGG16
# - VGG19
# - ResNet50
# - InceptionV3
# - InceptionResNetV2
# - MobileNet
# - DenseNet
# - NASNet
# - MobileNetV2

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Let's test one before theorizing

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.applications import InceptionV3

conv_base = InceptionV3(weights='imagenet', # Pre-trained on ImageNet data
                  include_top=False,        # Remove classification layer
                  input_shape=(28*3, 28*3, 1*3))  # IncpetionV3 requires at least 75x75 RGB
for layer in conv_base.layers:
    layer.trainable = False
conv_base.summary()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Expand network

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import Model

base_out = conv_base.output
base_out = Flatten()(base_out)
base_out = Dense(1024, activation='relu')(base_out)
base_out = Dropout(.5)(base_out)
base_out = Dense(10, activation='softmax')(base_out)
InceptionV3_model = Model(conv_base.input, base_out)

InceptionV3_model.compile(optimizer='adam',
          loss='categorical_crossentropy', 
          metrics=['accuracy'])

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Update data generator with new size

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=False)
datagen.fit(X_train_centered)
train_generator = datagen.flow(np.array(X_train_centered.repeat(3,1).repeat(3,2).repeat(3,3)), np.array(Y_train), 
                               batch_size=64)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Train InceptionV3 on MNIST
# ~ 1h 30m on teacher's notebook

# + {"slideshow": {"slide_type": "-"}}
historyFlowInceptionV3 = InceptionV3_model.fit_generator(
    train_generator,
    epochs=10, steps_per_epoch=len(X_train_centered) / 64,
    validation_data=(np.array(X_valid_centered.repeat(3,1).repeat(3,2).repeat(3,3)), np.array(Y_valid)), 
    validation_steps=len(X_valid_centered) / 64)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Architectures
# - Non-sequential network topology
#     - Connections between non-neighbour layers (residual networks, skip-connections)
#         - Possibly to all later layers (DenseNet)
#     - Parallell filter groups, e.g. series of convoutions in parallel (e.g. Inception cells)
#     - Extra input and/or output layers in the network
# - Can be modular - blocks of convolutions used several times
# - Keras Applications: pre-built, pre-trained
# - Implementation requires `keras.Model` (Functional API)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Keras' Functional API

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,)) # , dtype='int32', name='main_input')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)  # starts training

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Residual networks
# Add activation from a layer to the pre-activation of a later layer.
# <img src="./images/ResNet1.png" alt="Feature map" style="width: 400px;"/>

# + {"slideshow": {"slide_type": "slide"}}
# Add a shortcut/residual to a network (ResNet)
from tensorflow.keras.layers import ReLU, Add

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
shortcut = x           # Branch out

x = Dense(32, activation='relu')(x)
x = Dense(64)(x)       # No activation

x = Add()([shortcut, x]) # Add outputs (Make sure sizes match up)
x = ReLU()(x)
predictions = Dense(10, activation='softmax')(x)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Implementation in Applications:
# <img src="./images/ResNet.png" alt="Feature map" style="width: 800px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Inception networks
# - Inception cells / modules
#     - Extract features on different scales, concatenate output
#     - Use padding="same" to preserve sizes for concatenation
# - Usually combined with bottlenecks to reduce channel depth (number of parameters)
# <img src="./images/Inception_cell.png" alt="Inception cell" style="width: 600px;"/>

# + {"slideshow": {"slide_type": "slide"}}
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate

def inception_cell(x):
    # 1x1 convolution
    x1 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    # 1x1 + 3x3 convolution
    x3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x3)

    # 1x1 + 5x5 convolution
    x5 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x5 = Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x5)
    
    # MaxPool + 1x1 convolution
    xp = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding="same")(x)
    xp = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(xp)
 
    x = Concatenate()([x1, x3, x5, xp])
    return(x)
    
inputs = Input(shape=(28,28,1))

x = inception_cell(inputs)
x = inception_cell(x)

predictions = Dense(10, activation='softmax')(x)


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Inception V1
# <img src="./images/InceptionV1.png" alt="Inception V1" style="width: 900px;"/>
# ~5 million parameters

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Kernel stacking
# - Larger spatial filters are more expressive and able to extract features at a larger scale.
# - Stacking two 3x3 filters covers the same space as one 5x5 filter.
#     - 5x5xc = 25c parameters vs 2 x 3x3xc = 18c parameters
# - Stacking a 1x3 and a 3x1 filter covers the same space as one 3x3 filter.
#     - 9c vs 6c
# - Filter stacking increases focus in the centre of the larger filters.
# <img src="./images/Conv_5x5_3x3.png" alt="Inception cell" style="width: 500px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Kernel stacking
# - Most accurate representation with linear (no) activation.
# - Best performance activating (e.g. ReLU) between stacked layers too.

# + {"slideshow": {"slide_type": "-"}}
def Conv2D_stack_n_1(x, n):
    xn = Conv2D(filters=32, kernel_size=(n,1), strides=(1,1), padding="same", activation="relu")(x)
    xn = Conv2D(filters=32, kernel_size=(1,n), strides=(1,1), padding="same", activation="relu")(xn)
    return(xn)


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Batch normalization
# - Rescale activations to maintain mean and standard deviation close to 0 and 1.
# - Especially useful for networks with many layers.
# - Not fully understood why it has a positive effect.

# + {"slideshow": {"slide_type": "-"}}
from tensorflow.keras.layers import BatchNormalization

inputs = Input(shape=(28,28,1))

x = inception_cell(inputs)
x = BatchNormalization()(x)
x = inception_cell(x)
x = BatchNormalization()(x)
x = inception_cell(x)
x = BatchNormalization()(x)
x = inception_cell(x)
x = BatchNormalization()(x)

predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/Architectures.jpg" alt="Architectures" style="width: 900px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Optimization
# ### Momentum
# - The _classical momentum_ method (Polyak, 1964) is a technique for accelerating gradient descent that accumulates a velocity vector in directions of persistent reduction in the objective across iterations.
#     - "Successful updates of weights are given an extra push."
#     - Momentum affects convergence most in the "transient phase", i.e. before fine tuning.
# - _Nesterov momentum_ adds a partial update of the gradient before adding momentum to the weight update.
#     - "Successful updates of weights are given two extra pushes."
# - Explanation and Python examples: https://towardsdatascience.com/a-bit-beyond-gradient-descent-mini-batch-momentum-and-some-dude-named-yuri-nesterov-a3640f9e496b

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Decay
# - Where momentum accellerates in promising directions, decay reduces the learning rate.
# - Finetuning of models requires small steps.
#     - Shrink the learning rate gradually
#     - For instance $lr_{epoch ~ i+1} = lr_{epoch ~ i} \times 0.99$
# - Too fast decay => never reach minimum

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Optimizers
# - AdaGrad: Parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. 
#     - The more updates a parameter receives, the smaller the updates.
#     - AdaDelta: Same idea, but in a local window of iterations.
# - RMSProp: Divide the gradient/learning rate for each weight by a running average of their recent, individual magnitudes - [Lecture by G.F.Hinton](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Optimizers
# - Adam is a robust gradient-based optimization method inspired by RMSProp and AdaGrad,
#     - suited for nonconvex optimization and machine learning problems,
#     - choice of update step size derived from from the running average of gradient moments.
#     - Used to have guaranteed convergence, but now reduced to practically always converges.
# - SGD with momentum also popular (no invidual learning rates)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/Optimization1.gif" alt="Optimization" style="width: 600px;"/>
# Image credit [Alec Radford](https://twitter.com/alecrad)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <img src="./images/Optimization2.gif" alt="Optimization" style="width: 600px;"/>
# Image credit [Alec Radford](https://twitter.com/alecrad)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Initialization of weights
# - Can be important for convergence.
# - Variations of truncated normal distributions are often used (> +/-2 stddev are redrawn).
# - Dense and Conv2D have the Glorut normal initializer as default:
#     - Truncated normal distribution with stddev = sqrt(2 / (fan_in + fan_out)), where fan_in is the number of input units in the weight tensor and  fan_out is the number of output units in the weight tensor.
# - Biases are usually initialized as 0s.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## More regularization
# - We have regularized the network through dropout.
# - L1 and/or L2 norm regularization can be added to kernels, biases and activatitions separately for each layer.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Back propagation
# - Short version: "Same principle, just accumulated in each 'pixel' for all strides of the kernel".
#     - Errors are propagated backward for local gradient updates, but these now apply to all the positions a kernel can take over the previous layer's feature map.
# - More detailed explanations, illustrations and example code: [Back Propagation in Convolutional Neural Networks — Intuition and Code](https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Summary

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - Convolution: filters applied to all sub-patches of images
# - Padding: full, same, valid
# - Sub-sampling: max pooling, average pooling
# - 4D tensors for 2D images ($\#images \times \#pixel_x \times \#pixel_y \times \#channels$)
# - Dropout: Reducing capacity, increasing redundancy

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - Sequential models in Keras
# - TensorBoard: Visualisation, real-time inspection
# - Plotting: Interpretation of layer effects
# - Image augmentation: ImageGenerator for random image changes in training
# - `.flow` and `.flow_from_directory` for batch-wise processing

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - Architectures (Keras Applications)
#     - Used as _convolutional bases_ in Keras
#     - Some of their building blocks
# - Kernel stacking: Stack small kernels to achieve coverage of large kernels
# - Batch normalization: Normalize Activations for deep networks

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - Optimization
#     - Momentum: Accelerate convergence in promising directions
#     - Decay: Reduce learning rate over epochs
# - Initialization of weights: Keep default unless you know what you're doing.
# - Regularization: L1, L2 or both on individual layers. Oliver explains.
# - Back propagation: Same principle, repeated over all kernel positions

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
#   
#   
#   
# ### Easy, peasy ...
