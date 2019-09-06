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
 
 Use the full dataset to predict the response (ordinary linear regression
 should suffice, but other tools may work). Apply your feature engineering
 tools and try to achieve predictions that are less than 10^-4 off target.
 During the exploration, visualise and ponder on the attributes of the
 available features.
 
 Data are available as a CSV file with latin1 encoding.
 """