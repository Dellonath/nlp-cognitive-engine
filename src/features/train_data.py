import os
import sys
import pandas as pd 
import pickle
from preprocessing import Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

class TrainData():

    def __init__(self):

        self.tfidf = TfidfVectorizer()

        # input and output data paths
        self.processed_path = 'data/processed/'
        self.train_path = 'data/train/'

    def tranform(self, text):
        with open('models/tfidf/tfidf_vectorizer.pickle', 'rb') as handle:
            self.tfidf = pickle.load(handle)
        
        return self.tfidf.transform([text]).toarray()

    def get_feature_names(self):
        return self.tfidf.get_feature_names_out()

    def make(self, data_path):

        data = self.__fit_tfidf(self.processed_path + data_path)

        with open('models/tfidf/tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        data = pd.DataFrame(data.toarray(), columns = self.get_feature_names())

        data.to_parquet(f'data/train/train.parquet', index = False, engine = 'pyarrow')
    
    def __fit_tfidf(self, data_path):
        data = self.__read_data(data_path)
        pd.get_dummies(data['intent']).to_parquet('data/train/target.parquet', index = False, engine = 'pyarrow')
        data.drop(columns = ['intent'], axis = 1, inplace = True)
        return self.tfidf.fit_transform(data.phrase_cleaned)
    
    def __read_data(self, data_path):
        return pd.read_parquet(data_path)

if __name__ == '__main__':
    td = TrainData()
