# -*- coding: utf8 -*-
import logging
import sqlite3

import gensim


def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words[0].split(' '), [i])

if __name__ == '__main__':

    # f = open('dataPost/WordsSpecLine.txt', 'rb')
    # data = pickle.load(f)
    # data = [item for sublist in data for item in sublist]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""select sent_text from sentences""")
    data = cursor.fetchall()

    train_data = list(create_tagged_document(data))
    print(train_data)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, negative=10, min_count=5)
    # model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=300, negative=10, min_count=5, alpha=0.065, min_alpha=0.065,  workers=multiprocessing.cpu_count())
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("models/w2v/model_full.doc2v")
