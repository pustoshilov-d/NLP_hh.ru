# -*- coding: utf-8 -*-
import json
import sqlite3

import requests

response = requests.get('https://api.hh.ru/specializations')
data = response.json()

comm_file = open('data/commonJSON.txt', 'w+')
json.dump(data, comm_file)
comm_file.close()

conn = sqlite3.connect("data.db")
cursor = conn.cursor()
cursor.execute("""drop TABLE sentences""")
cursor.execute("""CREATE TABLE sentences
                  (id INTEGER PRIMARY KEY, field_id INTEGER, field_name text, spec_id INTEGER, spec_name text, sent_text text)
               """)

k_Spec = 0
k_Field = 0

lim = 200
k = 0


print(data)
try:
    for field in data: #8
        fieldName = field['name']
        try:
            print(field)
            for spec in field['specializations']: #20
                specName = spec['name']
                try:
                    # print(spec)
                    id_spec = spec['id']
                    print(id_spec)
                    # if int(id_spec.split('.')[0]) not in [1,20,17,13,5,21,18]:
                    #     print(True)
                    spec_file = open('data/' + id_spec + '.txt', 'w+')
                    for n_page in range(20):
                        try:

                            responseInfo = requests.get('https://api.hh.ru/vacancies?specialization=' + id_spec + '&page=' + str(n_page) + '&per_page=100')
                            # print(responseInfo.encoding)
                            info = responseInfo.json()
                            for specInfo in info['items']:
                                # print(specInfo)
                                try:

                                    spec_file.write(' '.join([str(specInfo['name']), str(specInfo['snippet']['requirement']), str(specInfo['snippet']['responsibility'])])+'\n')
                                    # line = ' '.join([str(specInfo['name']), str(specInfo['snippet']['requirement']), str(specInfo['snippet']['responsibility'])])
                                    # line = ' '.join(review_to_wordlist(line))
                                    # sentence = [(str(k_Field), fieldName, str(k_Spec), specName, line)]
                                    # print(sentence)
                                    # cursor.executemany(
                                    #     "INSERT INTO sentences(field_id, field_name, spec_id, spec_name, sent_text) VALUES (?, ?, ?, ?, ?)",
                                    #     sentence)
                                    # conn.commit()

                                except Exception as e: print(e)



                        except Exception as e: print(e)

                        spec_file.close()

                except Exception as e: print(e)
            k_Spec += 1
        except Exception as e: print(e)
    k_Field += 1
except Exception as e: print(e)
