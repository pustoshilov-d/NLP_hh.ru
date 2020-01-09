# -*- coding: utf8 -*-

import os
# import DataPreprocessing
import pickle
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':

    os.environ["PATH"] += os.pathsep + 'D:\Programing\\bin\\'
    logdir = "logs\\3\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    history = None

    X = pickle.load(open('dataFin/X.txt', 'rb'))

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""select field_id from sentences""")
    # cursor.execute("""select field_id from sentences""")
    rec = cursor.fetchall()
    Y = [i[0] for i in rec]
    print()

    X, Y = shuffle(X,Y)
    num_classes = len(np.unique(Y))
    # X = X[:120000]
    # Y = Y[:120000]
    # print(X[0], Y[0])

    batch_size = 10000

    num_samples = len(X)
    input_shape = X[0].size
    epochs = 100
    num_layers = 2
    print(num_classes,num_samples, input_shape)

    Y = to_categorical(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size=0.7)

    #веса правилом геометрической пирамиды

    model = Sequential()

    if num_layers == 2:
        r = pow(input_shape / num_classes, 1/3)
        k1 = round(num_classes * pow(r,2))
        k2 = round(num_classes * r)
        print('Скрытых слоёв: ', num_layers, '. Нейронов скрылых слоёв: ', k1, k2)

        model.add(Dense(k1, input_dim=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(k2))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

    if num_layers == 1:
        k = round(pow(input_shape * num_classes, 1/2))
        # k = 128
        print('Скрытых слоёв: ', num_layers, '. Нейронов скрылых слоёв: ', k)
        model.add(Dense(k, input_dim=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))


    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    plot_model(model, to_file='models/CNN/modelD2V_good.png', show_shapes=True)

    callbacks = [
        ModelCheckpoint('models\\CNN\\checkpoints\\modelD2V_best.h5', save_best_only=True),
        TensorBoard(log_dir=logdir, histogram_freq=1),
        EarlyStopping(monitor='val_loss')
    ]
    #tensorboard --logdir D:\Programing\Projects\course_work_ISY7\logs\3

    history = model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks, validation_split=0.2, shuffle=True)

    # model.save('models\\CNN\\modelD2V_full.h5',overwrite=True)

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test accuracy:', score[1])

    plt.plot(history.history['accuracy'], label = 'Аккуратность на обуч')
    plt.plot(history.history['val_accuracy'], label = 'Аккуратность на трень')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Аккуратность')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label = 'Ошибка на обуч')
    plt.plot(history.history['val_loss'], label = 'Ошибка на трень')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()


