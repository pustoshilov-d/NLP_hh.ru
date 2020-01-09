# -*- coding: utf8 -*-
import json
import pickle
import re

import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords


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



    dataSplited = []
    dataAll = []

    columns = ['fieldId', 'fieldName', 'specId', 'specName', 'sentText']
    sentences = pd.DataFrame(columns=columns)

    k_Spec = 0
    k_Field = 0

    lim = 5
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
            for lines in currFile.readlines():
                # if k == lim: break

                rev_line = review_to_wordlist(lines)
                dataAll += rev_line
                dataSpec.append(rev_line)

                sentences = sentences.append(pd.DataFrame(
                    [[k_Field, fieldName, k_Spec, specName, ' '.join(rev_line)]],
                    columns=columns),ignore_index=True)
                print('k ', k)
                k += 1

            dataSplited.append(dataSpec)
            k_Spec += 1
        k_Field += 1

    dataAll = np.array(dataAll)

    print(dataAll.shape)
    print(len(dataSplited))
    print(sentences)
    print(sentences.describe()['fieldId'], sentences.describe()['specId'])


    file = open('dataPost/WordsSpecLine.txt', 'wb+')
    pickle.dump(dataSplited, file)
    file.close()

    file = open('dataPost/Sentences.txt', 'wb+')
    pickle.dump(sentences, file)
    file.close()






