# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
import re 
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class Type_Classification(object): 

    def __init__(self, train_df, model, language):
        self.language = language
        self.train_df = train_df
        self.tfidf = TfidfVectorizer(sublinear_tf=True,
                            analyzer='word', ngram_range=(1, 4), 
                            min_df = 1, stop_words='english',norm='l2')
        self.model = model
        self.lemmatizer = WordNetLemmatizer()
    
    def text_basic_clean(self, text):

        text = text.replace('\n\n•', '').replace('\n\n', '')
        text = re.sub(r'[^\w\s]', '', text) 
        text = text.replace('  ', ' ')
        if self.language == 'english':
            text = ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in text.split() if word not in stopwords.words('english') and word.isalpha()])
        elif self.language == 'french':
            text = ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in text.split() if word not in stopwords.words('french') and word.isalpha()])
        elif self.language == 'spanish':
            text = ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in text.split() if word not in stopwords.words('spanish') and word.isalpha()])
        return text 
    

    def label_mapping(self, label):
        if label == "NOT_RELEVANT":
            return 0
        elif label == "RELEVANT":
            return 1    

    def data_Preprocessing(self):
        df = self.train_df
        df = df[df.relevance != 'N_A']
        df['content2'] = df['content'].apply(lambda x: self.text_basic_clean(x)) 
        df['relevance'] = df['relevance'].apply(lambda x: self.label_mapping(x)) 
        return df

    def get_embeddings(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts).toarray()
        return tfidf_matrix

    def get_embedding_model(self):
        relevance_data = self.data_Preprocessing()
        self.tfidf.fit(list(relevance_data.content2))
        return self.tfidf

    def fit(self):
        relevance_data = self.data_Preprocessing()
        tfidf_matrix, y_train = self.get_embeddings(list(relevance_data.content2)), np.array(relevance_data.relevance)
        fitted_model = self.model.fit(tfidf_matrix, y_train)
        return fitted_model

    def predict_proba(self, test_text):
        model = self.fit()
        if type(test_text) == str:
            test_texts = [test_text]
        test_texts = [self.text_basic_clean(x) for x in test_texts]
        x_test = self.tfidf.transform(test_texts)
        prob = model.predict_proba(x_test)
        if type(test_text) == str:
            return prob[0]
        return prob
      

    def predict(self, test_text):
        model = self.fit()
        if type(test_text) == str:
            test_texts = [test_text]
        test_texts = [self.text_basic_clean(x) for x in test_texts]
        x_test = self.tfidf.transform(test_texts)
        label = model.predict(x_test)
        if type(test_text) == str:
            return label[0]
        return label
