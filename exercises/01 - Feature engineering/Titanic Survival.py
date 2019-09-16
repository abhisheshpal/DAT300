# -*- coding: utf-8 -*-
"""
Updated 2019.09.09

@author: Kristian Hovde Liland
"""

"""
Titanic survival

Test your Kaggle connection from Colab.
- Download titanic survival data (https://www.kaggle.com/c/titanic/data)
- Make dataset more complete
    - Apply OneHotEncoder
    - Apply SimpleImputer
    - More ...
- Make prediction of survival rate
"""


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
#import Dataframes as df
#import utils as utils

data_gender = pd.read_csv("gender_submission.csv") 
data_gender.head()

data_train = pd.read_csv("train.csv") 
data_train.head()

data_test = pd.read_csv("test.csv") 
data_test.head()

test_train = data_train.append(data_test, sort=False)
test_train.head()

test_train.columns(3)





lebe = LabelEncoder()
x_train = lebe.fit_transform(test_train)
x_train.head()

# +


class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):

        self.columns = columns # list of column to encode
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# -

le = MultiColumnLabelEncoder()
X_train_le = le.fit_transform(data_test)
X_train_le.head()

# +
#fem = ['female']
#mal = ['male']

#enc = preprocessing.OneHotEncoder(categories=[fem, mal])
# Note that for there are missing categorical values for the 2nd and 3rd
# feature
enc = preprocessing.OneHotEncoder()
data_train = [['male'], ['female']]
enc.fit(data_train) 

OneHotEncoder(categorical_features='Sex',
       categories=[...], drop=None,
       dtype=np.float64, handle_unknown='error',
       n_values=None, sparse=True)
enc.transform([['male']]).toarray()
# -

enc.categories_

data_train

# +
enc = preprocessing.OneHotEncoder()
X = [['male'], ['female']]
enc.fit(X)

OneHotEncoder(categorical_features=None, categories=None, drop=None,dtype= np.float64, handle_unknown='error',
       n_values=None, sparse=True)
enc.transform([['male'],
                ['female']]).toarray()
# -

enc.categories_

full_data=train_data.append(test_data)
