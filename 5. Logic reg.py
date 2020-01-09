# -*- coding: utf8 -*-
import multiprocessing
import pickle
import sqlite3

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

if __name__ == '__main__':

    flag = 'Simple'
    # flag = 'D2V'
    logreg = None
    X_train = X_test = y_train = y_test = None

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""select field_name, spec_name from sentences""")
    # cursor.execute("""select field_name from sentences""")
    rec = cursor.fetchall()
    # print(rec[0])
    Y = [i[0]+": "+i[1] for i in rec]
    # Y = [i[0] for i in rec]
    print(Y[0])




    if flag == 'Simple':

        cursor.execute("""select sent_text from sentences""")
        sentences = cursor.fetchall()
        X = [i[0] for i in sentences]
        print(X[0])

        X, Y = shuffle(X, Y)

        # X = X[:12000]
        # Y = Y[:12000]

        X, Y = shuffle(X,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,shuffle = True)

        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression(n_jobs=4, verbose=1)),
                           ])

        logreg.fit(X_train, y_train)
        sFile = open('models/other/logregSimple_full.h5', 'wb+')
        y_pred = logreg.predict(["react javascript разработка сайта"])
        print(y_pred)

    else:
        X = pickle.load(open('dataFin/X.txt', 'rb'))
        X, Y = shuffle(X,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle = True)

        logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1e5)
        logreg.fit(X_train, y_train)
        sFile = open('models/other/logregD2V.h5', 'wb+')


    pickle.dump(logreg, sFile)
    sFile.close()

    y_pred = logreg.predict(X_test)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))






