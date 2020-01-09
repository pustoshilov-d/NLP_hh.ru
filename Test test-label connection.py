# -*- coding: utf8 -*-
import pickle

import numpy as np
from gensim.models import Doc2Vec

from Preprocessing_Text_1 import review_to_wordlist


def eval(x, strVecAll):
    x = np.array(x)
    strVecAll = np.array(strVecAll)
    predictions = pow(np.linalg.norm(x-strVecAll, axis=1),2)
    predictions = np.array(predictions)
    y= predictions.argmin()
    return y



if __name__ == '__main__':

    # x = np.array([1,2])
    # Y = np.array([[1,1],[2,2],[3,3]])
    # print(np.linalg.norm(Y-x, axis=1))

    modelD2V = Doc2Vec.load('models/w2v/model.doc2v')

    # str1 = "дизайн сайта"
    str1 = "react javascript разработка сайта программист"
    strVec1 = modelD2V.infer_vector(review_to_wordlist(str1))

    data = pickle.load(open('dataPost/Sentences.txt', 'rb'))
    strVecAll = []
    specNames = []
    for name in (data['fieldName'] + " " + data['specName']).unique():
        print(name)
        str = modelD2V.infer_vector(review_to_wordlist(name))
        specNames.append(name)
        strVecAll.append(str)
    print(len(specNames))

    X = pickle.load(open('dataFin/X.txt', 'rb'))
    Y = data['specId']
    res = []
    k = 0
    for x in X:
        res.append(Y == eval(x, strVecAll))
        print(k)
        k += 1
    print(np.array(res).mean())



