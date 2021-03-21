# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 08:10:04 2021

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

# %% 6-27 Using the LSTM layer in Keras

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

embedding_dim = 100 

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

model.save_weights('lstm_no_glove_model.h5')

# %% Visualization
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

# En epoch = 7 tiene una acc = 97 %, val_acc = 87 % pero el mínimo de la
# val_loss = 0.31 está en epoch = 3  cuando loss = 0.22
# O sea, que tiene bias razonable y poco overfitting

# %% Tokenizing the data of the test set

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
                
# %% Evaluating the model on the test set
model.load_weights('lstm_no_glove_model.h5')
model.evaluate(x_test, y_test)

# test_acc = .8, test_loss = 0.68
# Queda bastante bias y hay mucho sobreajuste

# %% Compliquemos la red a ver si mejora
model_2 = Sequential()
model_2.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model_2.add(LSTM(32, return_sequences=True))
model_2.add(LSTM(32, return_sequences=True))
model_2.add(LSTM(32))
model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history_2 = model_2.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

model_2.save_weights('lstm_2_no_glove_model.h5')

acc = history_2.history['acc']
val_acc = history_2.history['val_acc']
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
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

# epoch = 4, val_acc = 0.87, acc = 0.94

model_2.load_weights('lstm_2_no_glove_model.h5')
model_2.evaluate(x_test, y_test)

# loss: 0.5895 - acc: 0.8437
# Out[349]: [0.5895082950592041, 0.843720018863678]