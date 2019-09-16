# -*- coding: utf-8 -*-
"""
Updated 2019.09.05

@author: Kristian Hovde Liland
"""

"""
 As the world leading producer of Flubby Flops (TM) you want to predict
 the flubberiness from the raw material analyses and input settings of 
 the Flubmaster EX. The following features are available:
 + Raw materials
     - flostard
     - flüber
     - lard
 + Process attributes
     - floppiness
     - process start
     - boiling stop
     - stretching stop
 + Response
     - flubberiness
 
 You have been introduced to several techniques in feature engineering in 
 your days at the university with regard to deriving new features, 
 recoding, transforming, and making interactions. ++
 
 Use the full dataset to predict the response, i.e. no validation or hold-out
 for this exercise (ordinary linear regression should suffice, but other tools 
 may work). Apply your feature engineering tools and try to achieve predictions
 that are less than 10^-3 off target.
 During the exploration, visualise and ponder on the attributes of the
 available features.
 
 Data are available as a CSV file with latin1 encoding.
 """

# # Steps in Data Preprocessing
# Step 1 : Import the libraries
#
# Step 2 : Import the data-set
#
# Step 3 : Check out the missing values
#
# Step 4 : See the Categorical Values
#
# Step 5 : Splitting the data-set into Training and Test Set
#
# Step 6 : Feature Scaling
#
#
# ![ML.png](attachment:ML.png)
# ![ml.jpeg](attachment:ml.jpeg)

import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from pandas import DataFrame
import csv

data = pd.read_csv('Flubby_Flops.csv', encoding = "ISO-8859-1")
data.head()

# +
# check the missing value

data.isnull().sum()    # seems 0 is missing in each column
data.shape
# mark all missing values
#data.replace('?', nan, inplace=True)


# +
#data = pd.read_csv('Flubby_Flops.csv', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'processstart':[0,1]}, index_col=['processstart'],encoding = "ISO-8859-1")

#data.head()

# +
# convert feature 'process_start', 'boiling_stop', 'stretch_stop', timestamp into numeric

#data_process = pd.DataFrame({'process_start': pd.date_range('2018-06-29 01:57:00', periods=987)})
#data
data['PS'] = pd.to_datetime(data.process_start)
data['BS'] = pd.to_datetime(data.boiling_stop)
data['SST'] = pd.to_datetime(data.stretch_stop)
data


# -

data.drop(columns=['process_start', 'boiling_stop', 'stretch_stop'], axis=1, inplace= True)
data.head()

#Drop unnecesary features from data and decide the features
data.drop(data.iloc[: , 0:1], axis=1, inplace= True)
data.head()

data.dtypes

# replace catagorical features (floppiness) with binary valaues
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

data.iloc[:,0] = label_enc.fit_transform(data.iloc[:,0])
data

dummy = pd.get_dummies(data.iloc[:,0])

dummy.head()


dummy.shape

data2 = pd.concat([data, dummy])

data2.head()

data3 = data.join([dummy])

data3.head()

data3 = data3.drop(columns=['floppiness'])

data3.head()


# +
# convert timestamp_to_float

def convert_to_float(a):
    return time.mktime(t.timetuple(a))


# -

from sklearn.linear_model import LinearRegression

X = data3.iloc[:,:] #independent columns


X = X.drop(columns=['flubberiness'])
X.head()

Y = data3.iloc[:,3] 

   #target column i.e price range
Y.head()

import seaborn as sns
#get correlations of each features in dataset
corrmat = X.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,8))
#plot heat map
g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")

model = LinearRegression()
model = model.fit(X,Y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# dummy_1 = pd.get_dummies(data_1['floppiness'])


# +
#dummy_1

# +
#data_new = pd.concat([data_1, dummy_1], axis=1)

# +
#data_new

# +
#data.drop(data.iloc[:, 17:20], axis=1)

# +
#data
# -







data = pd.DataFrame(data_flubby)
