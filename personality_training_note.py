# cancelled data due to complicated 3d RNN sampling
# nlp code for tokenizer 

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from hmmlearn import hmm


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
personality = json.load(open('personality.json').read())

words =[]
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for personality in personality['personality']:
    for pattern in personality['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append(word_list, personality['tag'])
        if personality['tag'] not in classes:
            classes.append(personality['tag'])

words = [lemmatizer.lemmatizer(word) for word in words if word not in ignore_letters]
words = sorted( set(words))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(words,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = documents[0]
    word_patterns = [lemmatizer.lemmatize(word.lower() for word in word_patterns)]
    for word in words:
        bag.append(i) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
train_z = list(training[:,2])

modelhmm = Sequential()
model = Sequential()

model.add(Dense(123, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(8.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD( lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical.crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epoch =200, batch_size=5, verbose=1)

modelhmm.add(Dense(64, input_shape=(len(train_y[0]),), activation='relu'))
modelhmm.add(Dropout(0.5))
modelhmm.add(Dense(32, input_shaper=train_z[0), activation='relu'))

modelhmm.hmm.GaussianHMM(n_components=3, covariance_type=2, n_iter=10, verbose=1).fit(np.array(train_x), np.array(train_y), epoch =200, batch_size=5)

sgd = SGD( lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical.crossentropy', optimizer=sgd, metrics=['accuracy'])
model.save('personality.model')
print('done')
