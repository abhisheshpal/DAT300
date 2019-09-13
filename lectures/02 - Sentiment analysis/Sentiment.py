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
# # Chapter 8 - Applying Machine Learning To Sentiment Analysis
# <center><img src="./images/sentiment.png" alt="Sentiment analysis" style="width: 400px;"/></center>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Overview

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - [Preparing the IMDb movie review data for text processing](#Preparing-the-IMDb-movie-review-data-for-text-processing)
#   - [Obtaining the IMDb movie review dataset](#Obtaining-the-IMDb-movie-review-dataset)
#   - [Preprocessing the movie dataset into more convenient format](#Preprocessing-the-movie-dataset-into-more-convenient-format)
# - [Introducing the bag-of-words model](#Introducing-the-bag-of-words-model)
#   - [Transforming words into feature vectors](#Transforming-words-into-feature-vectors)
#   - [Assessing word relevancy via term frequency-inverse document frequency](#Assessing-word-relevancy-via-term-frequency-inverse-document-frequency)
#   - [Cleaning text data](#Cleaning-text-data)
#   - [Processing documents into tokens](#Processing-documents-into-tokens)
# - [Training a logistic regression model for document classification](#Training-a-logistic-regression-model-for-document-classification)
# - [Working with bigger data – online algorithms and out-of-core learning](#Working-with-bigger-data-–-online-algorithms-and-out-of-core-learning)
# - [Topic modeling](#Topic-modeling)
#   - [Decomposing text documents with Latent Dirichlet Allocation](#Decomposing-text-documents-with-Latent-Dirichlet-Allocation)
#   - [Latent Dirichlet Allocation with scikit-learn](#Latent-Dirichlet-Allocation-with-scikit-learn)
# - [Summary](#Summary)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Sentiment analysis
# - A subfield of Natural Language Processing (NLP)
# - Classify documents based on their polarity
#     - the attitude of the writer
#     - sometimes called "opinion mining"
# - Example from the Internet Movie Database (IMDb)
#     - 50000 movie reviews
#     - Predictor for positive and negative reviews <6 / >=6 stars (out of 10)
# - Similar examples with discussion fora
#     - Predictor for ideas and non-ideas (Lego, beer brewing, ...)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Topics
# - Cleaning and preparing text data
# - Building feature vectors from text documents
# - Training a machine learning model to classify positive and negative movie reviews
# - Working with large text datasets using out-of-core learning
# - Inferring topics from document collections for categorization

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Preparing the IMDb movie review data for text processing 

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# ## Obtaining the IMDb movie review dataset

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# The IMDb movie review set can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
# After downloading the dataset, decompress the files.
#
# 0) Use the code in the following cells to retreive and extact automatically.
#
# A) If you are working with Linux or MacOS X, open a new terminal window `cd` into the download directory and execute 
#
# `tar -zxf aclImdb_v1.tar.gz`
#
# B) If you are working with Windows, download an archiver such as [7Zip](http://www.7-zip.org) to extract the files from the download archive.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# **Optional code to download and unzip the dataset via Python:**

# + {"slideshow": {"slide_type": "-"}}
import os
import sys
import tarfile
import time


source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    if duration == 0:
        duration = 10**-3
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size
    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()


# + {"slideshow": {"slide_type": "slide"}}
# This download takes a couple of seconds at NMBU (<30)
if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
    
    if (sys.version_info < (3, 0)):
        import urllib
        urllib.urlretrieve(source, target, reporthook)
    
    else:
        import urllib.request
        urllib.request.urlretrieve(source, target, reporthook)

# + {"slideshow": {"slide_type": "fragment"}}
# The extraction can take several minutes as all 50,000 reviews are stored as separate text files
# (101,111 files). 
# Extracting to a synced folder (Dropbox, Google Drive, OneDrive, ...) may slow the process further.
if not os.path.isdir('aclImdb'):

    with tarfile.open(target, 'r:gz') as tar:
        tar.extractall()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Preprocessing the movie dataset into more convenient format
# Read all review files and append them sequentially into a Pandas dataframe.

# + {"slideshow": {"slide_type": "-"}}
import pyprind       # pip install pyprind, if you haven't used it before
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Shuffling the DataFrame
# - The data were read systematically: test, train; pos, neg.
# - Shuffling before storage means we can stream the data from file and obtain a random flow of reviews

# + {"slideshow": {"slide_type": "-"}}
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Save the assembled data as CSV file
# We will later be streaming from this file

# + {"slideshow": {"slide_type": "-"}}
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# + {"slideshow": {"slide_type": "-"}}
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

# + {"slideshow": {"slide_type": "-"}}
df.shape

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# <hr>
# ### Note
#
# If you have problems with creating the `movie_data.csv`, you can find a download a zip archive at 
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/tree/master/code/ch08/
# <hr>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Introducing the bag-of-words model

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - Represent documents as counts of words
# - Vocabulary across all documents
# - Sparse representation
#     - Only part of the vocabulary used in each text
# - Many ways to implement this:
#     - Potential for crazy overhead
#     - Previously mentioned hashing
#     - Here: scikit-learn CountVectorizer

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Transforming documents into feature vectors

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# By calling the fit_transform method on CountVectorizer, we will construct the vocabulary of the bag-of-words model and transform the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
#

# + {"slideshow": {"slide_type": "slide"}}
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

# + {"slideshow": {"slide_type": "fragment"}}
# Vocabulary with ordering (as dictionary)
print(count.vocabulary_)
print(sorted(count.vocabulary_))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Per document, array representation of the counts below as feature vectors. The values in the feature vectors are also called the raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*.

# + {"slideshow": {"slide_type": "-"}}
print(type(bag))
print(bag.shape)

# + {"slideshow": {"slide_type": "-"}}
print(bag.toarray())

# + {"slideshow": {"slide_type": "-"}}
print(sorted(count.vocabulary_))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### n-grams and K-mers
# - Single word counts => 1-gram (what we did above)
# - Counts of word sequences
#     - 2-gram: "lazy student", "student invents", "invents procrastinator(tm)"
#     - 3-gram: "bad ass teacher", "ass teacher flunks", "teacher flunks student"
# - Spam filters showed good performance with 3-grams and 4-grams (in 2007)
# - Parameter to the CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# - Nucleotides: ACGTACGAGATTC
#     - 3-mers: ACG, CGT, GTA, ....

# + {"slideshow": {"slide_type": "slide"}}
count2 = CountVectorizer(ngram_range=[2,2])
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag2 = count2.fit_transform(docs)

# + {"slideshow": {"slide_type": "-"}}
print(sorted(count2.vocabulary_))
print(bag2.toarray())

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Assessing word relevancy via term frequency-inverse document frequency

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - Words occuring frequently in multiple documents from both/all classes should be downweighted
#     - term frequency-inverse document frequency (tf-idf)
#
# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$
#
# - tf(t, d): term frequency
# - *idf(t, d)*: inverse document frequency

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# - *idf(t, d)*: inverse document frequency:
#
# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$
#
# - $n_d$ = #documents, *df(d, t)* is the number of documents *d* that contain the term *t*
# - optional 1 in denominator (omni-present words would get 0 without)
# - log is used to ensure that low document frequencies are not given too much weight

# + {"slideshow": {"slide_type": "slide"}}
# Transform tf to tf-idf:
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)

np.set_printoptions(precision=2)
print(bag.toarray())
print(sorted(count.vocabulary_))
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray()) # Word in many documents => less variation in tf-idf

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### scikit-learn implementation of tf-idf
# - differs a bit from the text book version
# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
#   
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
#
# - tfs are often normalized before computing tf-idfs, while scikit-learn normalizes tf-idfs instead (L2 per document):
# $$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
#
# - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Example using "is"
# - $tf("is") = 3$, in document 3
# - $n_d("is") = 3$ (#documents with "is")
#
# $$\text{idf}("is", d3) \left( = log\frac{1 + n_d}{1 + \text{df}(d, t)} \right) = log \frac{1+3}{1+3} = 0$$
#   
# - before normalization:
#
# $$\text{tf-idf}("is",d3) \left(  = \text{tf}(t,d) \times (\text{idf}(t,d)+1) \right) = 3 \times (0+1) = 3$$
#   
# - full tf-idf(3rd document): [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# $$\text{tf-idf}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2, 3.0^2, 3.39^2, 1.29^2, 1.29^2, 1.29^2, 2.0^2 , 1.69^2, 1.29^2]}}$$
#
# $$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$
#   
# $$ \left( ['and', 'is', 'one', 'shining', 'sun', 'sweet', 'the', 'two', 'weather'] \right)$$
#
# $$\Rightarrow \text{tf-idf}_{norm}("is", d3) = 0.45$$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Cleaning text data

# + {"slideshow": {"slide_type": "-"}}
df.loc[3, 'review'][-70:]

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# - We want to remove HTML tags and punctuation (retaining smileys).
# - This should be done before generating the bag-of-words

# + {"slideshow": {"slide_type": "slide"}}
import re
def preprocessor(text):
    # Regular expression for HTML tags
    text = re.sub('<[^>]*>', '', text)
    
    # Most typical emoticons (smileys)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    
    # Remove all non-word characters, convert to lower-case and add possible emoticons to the end.
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# + {"slideshow": {"slide_type": "slide"}}
# Effect of preprocessor on example
preprocessor(df.loc[3, 'review'][-70:])

# + {"slideshow": {"slide_type": "-"}}
df.loc[4, 'review'][-70:]

# + {"slideshow": {"slide_type": "fragment"}}
# Synthetic example:
preprocessor("</a>This :) is :( a test :-)!")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Apply the preprocessor

# + {"slideshow": {"slide_type": "-"}}
# This takes a few seconds
df['review'] = df['review'].apply(preprocessor)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Processing documents into tokens
# - Raw text can be converted to words in several ways
#     - Basic: Splitting at blank spaces
# - Often useful to remove variations of a word
#     - Word stemming looks for the stem of a word
#     - Porter stemmer (published 1979/80) still used a lot
#     - Snowball stemmer (Porter2/English), Lancaster (Paice/Husk) are faster, but more aggressive
#     - Part of the Natural Language Toolkit (conda install nltk / pip install nltk)

# + {"slideshow": {"slide_type": "slide"}}
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

# Define basic tokenizer and Porter stemmer version
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# + {"slideshow": {"slide_type": "fragment"}}
tokenizer('runners like running and thus they run')

# + {"slideshow": {"slide_type": "fragment"}}
tokenizer_porter('runners like running and thus they run')

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Stop-words
# - Some words are so common, they are usually removed before analysis
#     - is, and, has, like, ...
#     - 127 such in NLTK library
#     - tf-idfs are robust against stop words
# - Some types of language processing need the stop words too.

# + {"slideshow": {"slide_type": "slide"}}
import nltk

# Update to most resent stop-words
nltk.download('stopwords')

# + {"slideshow": {"slide_type": "fragment"}}
from nltk.corpus import stopwords

# Combine tokenizer with Porter stemmer and stop-word removal
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')
if w not in stop]

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Training a logistic regression model for document classification
# - We will train an LR classifyer on half the reviews and test on the remaining 25,000.
# - Preprocessing of HTML was done earlier
#     - including lower-case conversion and emoticon handling.
# - Use GridSearch to test the effect of stemming, stop-words, L1/L2 and C-parameter

# + {"slideshow": {"slide_type": "fragment"}}
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# + {"slideshow": {"slide_type": "skip"}}
# This code worked in 2018.
# Now the stop words are handled differently, hence a new version below.
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# TfidfVectorizer combines CountVectorizer and TfidTransformer with a single function.
tfidf = TfidfVectorizer(strip_accents=None, # Already preprocessed
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None], # Not this time, but use idf with normalization
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None], # Not this time
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],       # Raw counts without normalization 
               'vect__norm':[None],           # --------------||----------------
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='saga'))])
# Solver specified to silence warning and to enable l1 regularization

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=1) # Number of jobs different from 1 sometimes crashes on Windows.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Tokenizing of stop words
# - In the newest scikit-learn (as of 2019.09.12), stop words need to be preprocessed before entering the TfidfVectorizer

# + {"slideshow": {"slide_type": "-"}}
stops = []
for s in stop:
    stops.append(tokenizer(s)[0])
stopsPorter = []
for s in stop:
    stopsPorter.append(tokenizer_porter(s)[0])

# + {"slideshow": {"slide_type": "slide"}}
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# TfidfVectorizer combines CountVectorizer and TfidTransformer with a single function.
tfidf = TfidfVectorizer(strip_accents=None, # Already preprocessed
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stops, None], # Not this time, but use idf with normalization
               'vect__tokenizer': [tokenizer],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stops, None], # Not this time
               'vect__tokenizer': [tokenizer],
               'vect__use_idf':[False],       # Raw counts without normalization 
               'vect__norm':[None],           # --------------||----------------
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
             {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stopsPorter, None], # Not this time, but use idf with normalization
               'vect__tokenizer': [tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stopsPorter, None], # Not this time
               'vect__tokenizer': [tokenizer_porter],
               'vect__use_idf':[False],       # Raw counts without normalization 
               'vect__norm':[None],           # --------------||----------------
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='saga'))])
# Solver specified to silence warning and to enable l1 regularization

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=1) # Number of jobs different from 1 sometimes crashes on Windows.

# + {"slideshow": {"slide_type": "slide"}}
# The fitting of 2*2*2*3*5*2 models took around 30-60 minutes to fit in 2018. In 2019 it takes several hours. :(.
# Lowering the number of samples or parameters will make it quicker, but may reduce the performance greatly.
gs_lr_tfidf.fit(X_train, y_train)

# + {"slideshow": {"slide_type": "fragment"}}
# Pickle (store to disk) the Grid Search CV object
import pickle
with open('gs_lr_tfidf.pickle', 'wb') as f:
    pickle.dump(gs_lr_tfidf, f, pickle.HIGHEST_PROTOCOL)

# + {"slideshow": {"slide_type": "fragment"}}
# To open an object that has been pickled, you need to import the object's dependencies and local functions
import pickle
with open('gs_lr_tfidf.pickle', 'rb') as f:
    gs_lr_tfidf = pickle.load(f)

# + {"slideshow": {"slide_type": "fragment"}}
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

# + {"slideshow": {"slide_type": "fragment"}}
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Other options
# - Our choices of preprocessing, counting, etc. were not tested in all possible variations
# - Logistic Regression was tested, but
# - Naïve Bayes is popular for text classification
#     - Good performance on small datasets
#     - Variants used for K-mer classifications of nucleotides

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Working with bigger data - online algorithms and out-of-core learning
# - Large datasets require large resources
#     - Memory is often the first real bottleneck
# - **Out-of-core learning** works incrementally on smaller batches of a dataset
# - Stocastic gradient descent (SGD) updated incrementally
#     - Mini-batches also used in ANN lectures
# - partial_fit function from the SGDClassifier will be used to stream documents and train a logistic regression

# + {"slideshow": {"slide_type": "slide"}}
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2]) # Text without pos/neg, and pos/neg
            yield text, label


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Wait ..., what ..., "yield"???  
# https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do  
# Basically returns a generator instead of a result. The returned generator can be iterated over to "yield" each of the outputs of the loop once without generating the whole sequence in one go.

# + {"slideshow": {"slide_type": "slide"}}
# next() retrieves the next output of an iterator/generator
next(stream_docs(path='movie_data.csv'))


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Read documents from the stream
# - Use stream_docs and number of documents as input

# + {"slideshow": {"slide_type": "-"}}
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            # Accumulate texts and labels into lists
            text, label = next(doc_stream)
            docs.append(text) 
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Out-of-core limitations
# - Neither CountVectorizer, nor TfidVectorizer can work without the complete data set.
# - HashingVectorizer builds its "vocabulary" iteratively
#     - 32-bit MurmurHash3 function by Austin Appleby

# + {"slideshow": {"slide_type": "fragment"}}
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,    # Prepare for large variation in words
                         preprocessor=None,   # Included in ...
                         tokenizer=tokenizer) # the one defined previously

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Logistic regression
# - SGDClassifier with loss = "log" results in Logistic Regression
# - n_features = 2\*\*21 reduces chance of hash collisions, but increases number of features/coefficients in the LR 

# + {"slideshow": {"slide_type": "-"}}
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

if Version(sklearn_version) < '0.18':
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
else:
    clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

doc_stream = stream_docs(path='movie_data.csv')

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Running the out-of-core learner
# - Process 1000 documents in each batch

# + {"slideshow": {"slide_type": "-"}}
import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    # Process reviews
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    
    # Transform using the HashingVectorizer
    X_train = vect.transform(X_train)
    
    # Update the classifier
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Validation
# - Test performance on the last 5000 reviews
# - Not directly comparable to last CV segment above (10,000 reviews)

# + {"slideshow": {"slide_type": "-"}}
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# + {"slideshow": {"slide_type": "fragment"}}
# Use last 5000 for final update
clf = clf.partial_fit(X_test, y_test)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### word2vec
# - Google release in 2013
# - Unsupervised learning based on neural networks
# - Attempts to automatically learn the relationship between words
#     - Similar words in similar clusters
# - Can reproduce certain words using vector math
#     - Example: king - man + woman = queen

# + {"slideshow": {"slide_type": "notes"}, "cell_type": "markdown"}
# -----------------
# End of lecture 2019.09.12
# -----------------

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Topic modeling
# - Assign topics to unabelled documents
#     - For instance: sports, finance, world news, politics, local news, ...
#     - A clustering task (unsupervised learning)
# - Latent Dirichlet Allocation (LDA, another LDA), Blei et al. 2003

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Latent Dirichlet Allocation
# - Based on Bayesian inference
# - Generative probabilistic model looking for groups of words appearing frequently together across documents
# - Decomposes bag-of-words matrix into two matrices:
#     - a document to topic matrix, and
#     - a word to topic matrix,
#     - whose product yields aproximately the bag-of-words matrix
# - Number of topics is a hyperparameter
# - scikit-learn uses Expectation-Maximization for LDA fitting

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Latent Dirichlet Allocation with scikit-learn

# + {"slideshow": {"slide_type": "-"}}
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

# + {"slideshow": {"slide_type": "fragment"}}
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,         # Words that occur across too many documents are exluded
                        max_features=5000) # Most frequent words, limiting the dimensionality
                                           # Both can be tuned
X = count.fit_transform(df['review'].values)

# + {"slideshow": {"slide_type": "slide"}}
# This may take 5+ minutes to compute
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
# 'batch' uses all data in one go (most accurate), but slower than 'online' (online/mini-batch)
X_topics = lda.fit_transform(X)

# + {"slideshow": {"slide_type": "-"}}
lda.components_.shape

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Results of LDA
# - Print the 5 most important words for each of the 10 topics

# + {"slideshow": {"slide_type": "fragment"}}
n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
#     
# 1. Generally bad movies (not really a topic category)
# 2. Movies about families
# 3. War movies
# 4. Art movies
# 5. Crime movies
# 6. Horror movies
# 7. Comedies
# 8. Movies somehow related to TV shows
# 9. Movies based on books
# 10. Action movies

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### Examples from the results
# - Three reviews from the horror movie category (category 6 at index position 5):

# + {"slideshow": {"slide_type": "-"}}
horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Summary
# - bag-of-words, counts of words associated with a dictonary
#     - Reweighting from tf (term frequency) to tf-idf (term frequency-inverse document frequency)
#     - Normalizing
# - Clean input HTML-tags, dots, commas, other non-text elements
# - Tokenizing (split on blank spaces)
#     - Stemming, stop-words
# - Out-of-core processing of documents and streaming logistic regression
# - Latent Dirichlet Allocation
#     - Decompose bag-of-words into document-topic and word-topic matrices
