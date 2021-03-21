# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 07:31:46 2021

NLTK 
https://www.nltk.org/book

2. Accessing Text Corpora and Lexical Resources

@author: migue
"""

# %% 1   Accessing Text Corpora


# %% 1.1   Gutenberg Corpus
import nltk

nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)
