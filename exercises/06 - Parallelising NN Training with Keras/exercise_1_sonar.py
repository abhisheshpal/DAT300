# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:02:12 2019

@author: oliver
"""


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

