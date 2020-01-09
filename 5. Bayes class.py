# -*- coding: utf8 -*-

import sqlite3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

if __name__ == '__main__':
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""select field_name, spec_name from sentences""")
    # cursor.execute("""select field_name from sentences""")
    rec = cursor.fetchall()
    # print(rec[0])
    Y = [i[0]+": "+i[1] for i in rec]
    # Y = [i[0] for i in rec]
    print(Y[0])

    cursor.execute("""select sent_text from sentences""")
    sentences = cursor.fetchall()
    X = [i[0] for i in sentences]

    print(X[0])

    X, Y = shuffle(X,Y)
    #
    # X = X[:12000]
    # Y = Y[:12000]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(X_train, y_train)




    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
