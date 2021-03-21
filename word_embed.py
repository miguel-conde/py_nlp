# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:31:46 2021

WORD EMBEDDINGS WITH THE EMBEDDING LAYER

@author: migue
"""
#
# 6.6 - Loading the IMDB data for use with an Embedding layer
#
from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 10000
# Cuts off the text after this number of words (among the max_features most
# common words)
maxlen = 20

# Loads the data as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Turns the lists of integers into a 2D integer tensor of shape 
# (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#
# 6.7 - Using an Embedding layer and classifier on the IMDB data
#
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.embeddings import Embedding

model = Sequential()

# The Embedding layer takes at least two arguments: the number of possible 
# tokens (here, 10,000: 1 + maximum word index) and the dimensionality of the 
# embeddings (here, 8).
# The Embedding layer takes as input a 2D tensor of integers, of shape 
# (samples, sequence_length), where each entry is a sequence of integers.
# This layer returns a 3D floating-point tensor of shape 
# (samples, sequence_length, embedding_dimensionality).
# 
# Specifies the maximum input length to the Embedding layer so you can later 
# flatten the embedded inputs. After the Embedding layer, the activations have 
# shape (samples, maxlen, 8).
model.add(Embedding(10000, 8, input_length=maxlen, name = 'embed'))

# Flattens the 3D tensor of embeddings into a 2D tensor of shape 
# (samples, maxlen * 8)
model.add(Flatten())

# Adds the classifier on top
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = history.epoch 

#%%
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Extract embeddings
# https://stackoverflow.com/questions/51235118/how-to-get-word-vectors-from-keras-embedding-layer
embeddings = model.layers[0].get_weights()[0]
embeddings.shape
print(embeddings)