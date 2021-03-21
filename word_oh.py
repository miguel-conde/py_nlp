# -*- coding: utf-8 -*-
"""
6.1 - Word-level one-hot encoding (toy example)

Este es un archivo temporal.
"""

import numpy as np

# Initial data: one entry per sample (in this example, a sample is a sentence,
# but it could be an entire document)
samples = ['The cat sat on the mat.', 
           'The dog ate my homework.']

# Builds an index of all tokens in the data
token_index = {}

for sample in samples:
    # Tokenizes the samples via the split method. In real life, you’d also 
    # strip punctuation and special characters from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assigns a unique index to each unique word. Note that you don’t
            # attribute index 0 to anything.
            token_index[word] = len(token_index) + 1

# Vectorizes the samples. You’ll only consider the first max_length words in 
# each sample.
max_length = 10

# This is where you store the results.
results = np.zeros(shape=(len(samples), 
                          max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

#
# Listing 6.4 Word-level one-hot encoding with hashing trick (toy example)
#
dimensionality = 10000

results_hash = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results_hash[i, j, index] = 1.