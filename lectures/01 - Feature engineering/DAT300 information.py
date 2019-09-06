# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # DAT300

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# <center><img src="./images/DAT300_H2019.jpg" alt="DAT300" style="width: 736px;"/></center>

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - We are being watched (recorded)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Plan for September 5th
# - Master project teaser by Jon Nordby
# - Introduction to:
#   - People involved
#   - Resources and tools
#   - Book(s)
#   - Lesson plan
# - Introduction and hands on with Google Colab' and Kaggle API
# - Feature enineering

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The men
#
# | Oliver Tomic | Kristian Hovde Liland |  
# |:-------------------------|----------:|  
# | <img src="./images/Oliver.jpg" alt="Oliver" style="height: 250px;"/> | <img src="./images/Kristian.jpg" alt="Kristian" style="height: 220px;"/>    |  
#     
# Associate Professors in Data Science.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The boys
# | Runar Helin              | Stanislau "Stas" Trukhan  |Mike Riess |  
# |:-------------------------|:-------------------------:|----------:|  
# | <img src="./images/runar.jpeg" alt="Runar" style="height: 250px;"/> | <img src="./images/stas.png" alt="Stas" style="height: 250px;"/>    | <img src="./images/mike.png" alt="Mike" style="height: 250px;"/>    |  
#     
# PhD students applying/developing machine- and deep learning methods.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## The resources
# - Canvas: https://nmbu.instructure.com/courses/4151  
#   - Information hub
#   - Compulsory assignments
# - GitLab: https://gitlab.com/nmbu.no/emner/DAT300/h2019  
#   - Git source for lectures and exercises (please register)  
# - Google Colaboratory: https://colab.research.google.com  
#   - Online processing (please register)
# - Kaggle: https://www.kaggle.com/
#   - Compulsory assignments, competitions and solutions (please register)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## The tools
# - Python 3.7: https://www.anaconda.com/
# - Keras: https://keras.io/
#   - Deep learning interface to Tensorflow et al.
# - DASK: https://dask.org/
#   - Out-of-core machine learning
# - Jupytext: https://towardsdatascience.com/introducing-jupytext-9234fdff6c57
#   - Jupyter Notebook extension for dual editing - Python code and Jupyter Notebook
# - RISE slideshow: https://github.com/damianavila/RISE
#   - Jupyter Notebook extension for presentation mode!
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## The book
# <img src="./images/TheBook.jpg" alt="Python Machine Learning" style="width: 200px;"/>  
# - Python Machine Learning, 2nd edition, By Raschka & Mirjalili  
#     - Available in Boksmia
#     - .. or https://www.akademika.no/python-machine-learning/raschka-sebastian/mirjalili-vahid/9781787125933
#     - .. or https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition (also 3rd edition)
# - Material from Deep learning with Python, By Chollet: https://www.manning.com/books/deep-learning-with-python

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <center><img src="./images/Timetable.png" alt="Time table" style="width: 1400px;"/> </center>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <center><img src="./images/TentativePlan_2019.png" alt="Tentative plan" style="width: 1032px;"/></center>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Expected student background
# ### Programming (e.g. INF120 + INF200), Statistics (e.g. MATH-INF110/STAT100), Linear algebra (e.g. MATH113/MATH131)
# ### Machine Learning (e.g. DAT200)
# Python Machine Learning, Sebastian Raschka and Vahid Mirjalili, Packt Publishing
# - Chapter 1 – Giving Computers the Ability to Lear from Data 
# - Chapter 2 – Training Simple Machine Learning Algorithms for Classification
# - Chapter 3 – A Tour of Machine Learning Classifiers
# - Chapter 4 – Building Good Training Sets – Data Processing
# - Chapter 5 – Compressing Data via Dimensionality Reduction
# - Chapter 6 – Learning Best Practices for Model Evaluation and Hyperparamter Tuning 
# - Chapter 7 – Combining Different Models for Ensemble Learning
# - Chapter 10 – Predicting Continous Target Variables with Regression Analysis
# - Chapter 11 – Working with unlabelled data – Clustering Analysis

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <center><img src="./images/MarkZuckerberg.jpg" alt="Ethics" style="width: 650px;"/></center>
