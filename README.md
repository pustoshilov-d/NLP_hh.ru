# NLP_hh.ru
NLP classification of HR requests

Purpose of the project is realisation software for business recruitment.
Input: description of tasks that future employee should do in company. 
Output: best five specialisation titles for that. 

[There](https://github.com/pustoshilov-d/NLP_hh.ru/tree/master/models/idef0) are IDEF0 software models. 
1. Collecting database from hh.ru using its API: vacancy texts and specialisation titles. That's 28 fields * 20 specialisatins * 2000 vacancies = 680000 records.
2. Text database preprocessing.
3. Vector representation using Keras.Tokenizer and Gensim.Doc2Vec
4. Training models of CNN, Baiese classificator, Logistic regression.
5. Usage models application

[There](https://github.com/pustoshilov-d/NLP_hh.ru/tree/master/screen_shots) are results.
Summary models and database weight is about 3Gb so they did not uploaded there but could be shared opun contact with me: pustoshilov.d@gmail.com
