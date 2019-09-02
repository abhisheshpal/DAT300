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
# # Google Colaboratory
# ### Free$^+$ Jupyter notebook$^*$ with cloud computation and Google Drive integration$\dagger$
# <img src="./images/CO.png" alt="CO"/>  
# ### https://colab.research.google.com  
#
# $^+$ Up to around 12 hours continuous computations a day. Fresh session, start from scratch.  
# \* Mostly the same, some extensions, some limitations.  
# $\dagger$ Requires setup, not extremely quick. Faster interaction with Google Cloud Storage.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Why "Colab"?
#
# - Nice interface
# - Easy to install and run Linux and Python packages
# - Has load of stuff already installed
#     - For instance XGBoost, TensorFlow, Keras, ...
# - Move workload off your computer
#     - Ageing laptop?
#     - Long running computations?
# - GPU support: Tesla K80 (~2.5 TFLOPS DP vs KHL's laptop at ~8 GFLOPS DP)
# <img src="./images/K80.jpg" alt="K80"/>
#     - New GPUs anounced, but limited access

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Data files on Colab
# - Upload files from desktop or Google Drive
# - Direct file access on Google Drive is slow for large files
#     - Network connection to different servers
#     - Various ways of uploading files in API
# <img src="./images/Drive.png" alt="Google Drive" style="width: 300px;"/>

# + {"slideshow": {"slide_type": "slide"}}
example, git to drive, open from there

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Kaggle integration
# - Datasets download
# - Competition submission
# - https://github.com/Kaggle/kaggle-api
# - https://medium.com/@move37timm/using-kaggle-api-for-google-colaboratory-d18645f93648
# - DAT300's 'System stuff.ipynb'

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Seedbank
# - Community examples of Machine Learning with Colab
# - https://tools.google.com/seedbank/
# <img src="./images/seedbank.png" alt="Google Seedbank" style="width: 150px;"/>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # OpenML
# - \> 20 000 data sets
# - Workflows / data analysis flows
# - Comparison of ML methods on data sets, e.g. check expected performance
# - Download, analyse, upload for free
# <img src="./images/OpenML.png" alt="OpenML"/>
