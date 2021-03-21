# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:51:07 2021

6.1.3 - Putting it all together: from raw text to word embeddings

@author: migue
"""


# %% 6.8 Processing the labels of the raw IMDB data
#
import os

imdb_dir = os.path.join('.', 'data', 'aclImdb')
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# %% 6.9 Tokenizing the text of the raw IMDB data
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Cuts off reviews after 100 words
maxlen = 1000
# Trains on 200 samples
training_samples = 16000
# Validates on 10,000 samples
validation_samples = 10000
# Considers only the top 10,000 words in the dataset 
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Splits the data into a training set and a validation set, but first shuffles 
# the data, because you’re starting with data in which samples are ordered 
# (all negative first, then all positive)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#
# PREPROCESSING THE EMBEDDINGS
#

# %% 6.10 Parsing the GloVe word-embeddings file
#
glove_dir = os.path.join('.', 'glove6B' )

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# %% 6.11 Preparing the GloVe word-embeddings matrix
#

# Queremos una matriz con una fila por palabra (10 000) y cada fila es su 
# embedding vector - vector de dimensión 100
# El índice de la fila nos refiere a la palabra real mediante word_index
# 100 porque es la dimensión de los vectores en embedding_index
embedding_dim = 100 

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
            
# %% 6.12 Model definition
# 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, 
                    name = "embed"))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# %% 6.13 - Loading pretrained word embeddings into the Embedding layer
#
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# %% 6.14 Training and evaluation
# 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train, 
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')

# %% 6.15 Plotting the results
#
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#%% Visualization - acc
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

#%% Visualization - loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# %% 6-16 Training the same model without pretrained word embeddings
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('no_glove_model.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# En epoch = 2 tiene una acc = 91 %, val_acc = 88 % con mínimo de la
# val_loss = 0.3 cuando loss = 0.23
# O sea, que tiene bias razonable y poco overfitting

# %% 6-17 Tokenizing the data of the test set

test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
                
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
                
# %% 6-18 Evaluating the model on the test set
# model.load_weights('pre_trained_glove_model.h5')
model.load_weights('no_glove_model.h5')
model.evaluate(x_test, y_test)

# test_acc = .86, test_loss = 1.1
# Se confirman las conclusiones de validación
# Habría que volver a entrenar con solo dos épocas

# Para mejorar bias: cambiar arquitectura a una red recurrente