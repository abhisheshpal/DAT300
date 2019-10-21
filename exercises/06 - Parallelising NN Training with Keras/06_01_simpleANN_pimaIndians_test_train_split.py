# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:47:03 2019

@author: oliver
"""
# MLP with manual validation set

# =============================================================================
# Import needed modules
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np


# =============================================================================
# Set seed and load data
# =============================================================================
# fix random seed for reproducibility
seed = 7

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")


# =============================================================================
# Prepare dataset
# =============================================================================
# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=seed)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# =============================================================================
# Define, compile and train model
# =============================================================================
# Define model
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Fit the model
start = time.time()

model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=150, 
          batch_size=10)

# model.fit(X_train_std, y_train,  
#          epochs=150, 
#          batch_size=10)

stop = time.time()


# Alternative way of model evaluaton
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\n\ntest_acc:', test_acc)


# Print training time
print('Training time: {0}'.format(stop - start))


# Print model summary
model.summary()






