# -*- coding: utf8 -*-
import sqlite3

import numpy as np
import re
from nltk.corpus import stopwords
import pymorphy2
import json
import pickle
import pandas as pd
import numba


def review_to_wordlist(review):
    # import nltk
    # nltk.download('stopwords')
    morph = pymorphy2.MorphAnalyzer()
    stops = set(stopwords.words("english")) | set(stopwords.words("russian"))
    #1 удаление всех символов, кроме букв
    review_text = re.sub("[^а-яА-Яa-zA-Z]"," ", review)
    #2 нижний регистр и ё-е
    words = review_text.lower().replace('ё', 'е').split()
    #3 удаление стоп слов
    words = [w for w in words if not w in stops]
    #4 лемматизация (к нормальной форме)
    words = [morph.parse(w)[0].normal_form for w in words ]
    return(words)

if __name__ == '__main__':

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""drop TABLE sentences""")
    cursor.execute("""CREATE TABLE sentences
                      (id INTEGER PRIMARY KEY, field_id INTEGER, field_name text, spec_id INTEGER, spec_name text, sent_text text)
                   """)

    k_Spec = 0
    k_Field = 0

    # lim = 400
    k = 0

    commonFile = open('data/commonJSON.txt','r')
    commonData = json.loads(commonFile.read())

    for field in commonData:
        # if k == lim: break
        print(field)
        fieldName = field['name']

        for spec in field['specializations']:
            # if k == lim: break

            specName = spec['name']
            dataSpec = []

            print(k_Spec)
            currFile = open('data/' + spec['id'] + '.txt', 'r')
            sentences = []
            k_lines = 0
            for line in currFile.readlines():
                k_lines += 1
                k += 1

                rev_line = ' '.join(review_to_wordlist(line))

                sentences.append([str(k_Field), fieldName, str(k_Spec), specName, rev_line])
                print('k ', k)

                # if k_lines == lim: break
            print(sentences)

            cursor.executemany(
                "INSERT INTO sentences(field_id, field_name, spec_id, spec_name, sent_text) VALUES (?, ?, ?, ?, ?)", sentences)
            conn.commit()


            k_Spec += 1
        k_Field += 1





