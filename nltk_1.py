# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:36:15 2021

NLTK 
https://www.nltk.org/book

1. Language Processing and Python

@author: migue
"""

#%% 1 Computing with Language: Texts and Words

# %% 1.2   Getting Started with NLTK
import nltk

# nltk.download()

from nltk.book import *

text3

# %% 1.3   Searching Text

text1.concordance("monstrous")

text1.similar("monstrous")

text2.similar("monstrous")

text2.common_contexts(["monstrous", "very"])

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

text3.generate()

# %% 1.4   Counting Vocabulary
len(text3)

sorted(set(text3))

len(set(text3))

len(set(text3)) / len(text3)

text3.count("smote")

100 * text4.count('a') / len(text4)


def lexical_diversity(text): 
    return len(set(text)) / len(text)

def percentage(count, total): 
    return 100 * count / total

lexical_diversity(text3)

percentage(4, 5)

percentage(text4.count('a'), len(text4))

# %% 3   Computing with Language: Simple Statistics
saying = ['After', 'all', 'is', 'said', 'and', 'done',
          'more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]

# %% 3.1   Frequency Distributions
fdist1 = FreqDist(text1) 
print(fdist1)

fdist1.most_common(50)

fdist1['whale']

fdist1.plot(50, cumulative=True)

# %% 3.2   Fine-grained Selection of Words
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)

# %% 3.3   Collocations and Bigrams
list(bigrams(['more', 'is', 'said', 'than', 'done']))
list(bigrams(text3[0:10]))

# %% 3.4   Counting Other Things
[len(w) for w in text1] [1]

fdist = FreqDist(len(w) for w in text1)

print(fdist)

fdist.most_common()

fdist.max()

fdist[3]

fdist.freq(3)

# Functions Defined for NLTK's Frequency Distributions
# https://www.nltk.org/book/ch01.html#tab-freqdist

# %% 5   Automatic Natural Language Understanding

# %% 5.1   Word Sense Disambiguation