# -*- coding: utf8 -*-
import os
import sqlite3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import DataPreprocessing
import pickle
import numpy as np
from keras.models import load_model
from Preprocessing_Text_1 import review_to_wordlist
from gensim.models import Doc2Vec


def best_of_ (predictions, cursor, number = 5):

    predictions = np.array(predictions[0])
    args = (-predictions).argsort()
    res = []
    for i in range(number):
        cursor.execute("""select field_name, spec_name from sentences where spec_id=? limit 1""",(int(args[i]),))
        rec = cursor.fetchall()
        labelName =": ".join(rec[0])
        res.append([labelName, predictions[args[i]]*100])
    return res




if __name__ == '__main__':
    flag_1 = ''
    flag = 'CNN Simple'
    # flag = 'CNN D2V'
    # flag = 'LogReg Simple'
    # flag = 'LogReg D2V'

    # str = "обслуживанию медицинского оборудования"
    # str = "react javascript разработка сайта программист"
    # str = "Информационные технологии, интернет, телеком"
    # str = "Team Lead Developer (php) Как работаем: Опыт работы PHP-разработчиком от 5 лет. Понимание и стремление следовать принципам SOA, GRASP, KISS, SOLID, YAGNI, DRY. "
    str = input("Опишите, что вам нужно от профессионала: ")
    str = review_to_wordlist(str)
    print('Супер! Вам нужен специалист в сфере: \n(сфера, вероятность)')
    # data = pd.DataFrame
    # data = pickle.load(open('dataPost/Sentences.txt', 'rb'))

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    # Y = [i[0] for i in rec]

    if flag == 'CNN Simple':
        tokenize = pickle.load(open('models/CNN/modelTokenizer.h5', 'rb'))
        strVec = tokenize.texts_to_matrix([str])


        model = load_model('models\\CNN\\modelToken_full.h5')
        predictions = model.predict(strVec)

        res = best_of_(predictions, cursor, 5)
        for re in res:
            print(re)

    if flag == 'CNN D2V':
        modelD2V = Doc2Vec.load('models/w2v/model.doc2v')
        strVec = np.array(modelD2V.infer_vector(str))

        model = load_model('models\\CNN\\modelD2V_full.h5')
        predictions = model.predict([[strVec]])

        res = best_of_(predictions, cursor, 5)
        for re in res:
            print(re)

    if flag == 'LogReg Simple':
        str = " ".join(str)

        model = pickle.load(open('models\\other\\logregSimple.h5', 'rb'))
        y_predict = model.predict([str])

        print(y_predict)

    if flag == 'LogReg D2V':
        modelD2V = Doc2Vec.load('models/w2v/model.doc2v')
        strVec = np.array(modelD2V.infer_vector(str))

        model = pickle.load(open('models\\other\\logregD2V.h5', 'rb'))
        y_predict = model.predict([strVec])

        print(y_predict)