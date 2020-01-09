# -*- coding: utf8 -*-
import logging
import pickle
import sqlite3

import numpy as np
from gensim.models import Doc2Vec

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Doc2Vec.load('models/w2v/model.doc2v')
    # print(model['услуга'])
    # print(model.infer_vector(['услуга', 'sql', 'менеджер']))

    # sentences = pickle.load(open('dataPost/WordsSpecLine.txt', 'rb'))

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""select sent_text from sentences""")
    sentences = cursor.fetchall()

    X = []
    # Y = []

    k = 0
    lim = 4
    line_k = 0
    for line in sentences:
        X.append(model.infer_vector(line[0].split(' ')))
        # Y.append(label_iter)
        print(line_k)
        line_k += 1

    X = np.array(X)
    print(X.shape)
    # Y = np.array(Y)
    xFile = open('dataFin/X.txt','wb+')
    # yFile = open('dataFin/Y.txt','wb+')
    pickle.dump(X, xFile)
    # pickle.dump(Y, yFile)
    xFile.close()
    # yFile.close()

