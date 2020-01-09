# -*- coding: utf8 -*-
import os
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import text
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# X = pickle.load(open('dataFin/X.txt', 'rb'))
# rawData = pickle.load(open('dataPost/Sentences.txt', 'rb'))

conn = sqlite3.connect("data.db")
cursor = conn.cursor()
cursor.execute("""select field_id from sentences""")
rec = cursor.fetchall()
Y = [i[0] for i in rec]

print('y done')
cursor.execute("""select sent_text from sentences""")
sentences = cursor.fetchall()

X = [i[0] for i in sentences]

print('x done')

X, Y = shuffle(X, Y)
num_classes = len(np.unique(Y))
X = X[:100000]
Y = Y[:100000]


batch_size = 1000

num_samples = len(Y)

# input_shape = X[0].size
epochs = 100
num_layers = 2


train_sent,test_sent, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size=0.9)
print(len(X), len(train_sent))


max_words = 3000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_sent) # only fit on train
# pickle.dump(tokenize, open('models/CNN/modelTokenizer.h5', 'wb+'))
x_train = tokenize.texts_to_matrix(train_sent)
x_test = tokenize.texts_to_matrix(test_sent)

print(len(x_train),len(x_train[0]))

# encoder = LabelEncoder()
# encoder.fit(train_tags)
# y_train = encoder.transform(train_tags)
# y_test = encoder.transform(test_tags)
#
# num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(len(y_train[0]))



callbacks = [
    # ModelCheckpoint('models\\CNN\\checkpoints\\modelD2V_best.h5', save_best_only=True),
    # TensorBoard(log_dir=logdir, histogram_freq=1),
    EarlyStopping(monitor='val_loss')
]

model = Sequential()

if num_layers == 2:
    r = pow(max_words / num_classes, 1 / 3)
    k1 = round(num_classes * pow(r, 2))
    k2 = round(num_classes * r)
    print('Скрытых слоёв: ', num_layers, '. Нейронов скрылых слоёв: ', k1, k2)

    model.add(Dense(k1, input_dim=max_words))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(k2))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

if num_layers == 1:
    # k = round(pow(max_words * num_classes, 1 / 2))
    k = 128
    print('Скрытых слоёв: ', num_layers, '. Нейронов скрылых слоёв: ', k)
    model.add(Dense(k, input_dim=max_words))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
# model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

os.environ["PATH"] += os.pathsep + 'D:\Programing\\bin\\'
plot_model(model, to_file='models/CNN/modelSimple_bad.png', show_shapes=True)


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks,
                    )
# model.save('models\\CNN\\modelToken_full.h5', overwrite=True)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

plt.plot(history.history['accuracy'], label='Аккуратность на обуч')
plt.plot(history.history['val_accuracy'], label='Аккуратность на трень')
plt.xlabel('Эпоха обучения')
plt.ylabel('Аккуратность')
plt.legend()
plt.show()
#
# plt.plot(history.history['loss'], label='Ошибка на обуч')
# plt.plot(history.history['val_loss'], label='Ошибка на трень')
# plt.xlabel('Эпоха обучения')
# plt.ylabel('Ошибка')
# plt.legend()
# plt.show()