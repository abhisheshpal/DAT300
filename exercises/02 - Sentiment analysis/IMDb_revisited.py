# -*- coding: utf-8 -*-
"""
Updated 2019.09.13

@author: Kristian Hovde Liland
"""

"""
EXERCISE: Exchange Logistic regression with Naïve Bayes in sentiment analysis
(1) Use the code from the lecture to
   - Read the IMDb review dataframe
   - Run the preprocessor
   - Load the Pickled object (gs_lr_tfidf) from file
(2) Find the optimal combination of text processing from this object.
(3) Create a pipeline with the text processing steps, but add a 
(Gaussian) Naïve Bayes model at the end instead of logistic regression.
   - Fit the model on the training data
   - Check performance on test data
"""

"""
BONUS EXERCISE: Latent Dirichlet Allocation playground
Three parameters in the code from the lecture that will have a great impact on
the results:
    - max_df: maximum document frequency of words
    - max_features: maximum vocabulary size
    - n_components: number of topics

What happens to the topics if you reduce the number of components to 3?

Make an extreme case by reducing the vocabulary to a few words per topic, 
e.g. a vocabulary of 5 words x 3 topics = 15 (max_features). How does it
affect the resulting categories, and can you still interpret the 
accompanying words?
"""