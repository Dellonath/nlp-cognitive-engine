import os
import sys
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class PrepareTrainData():

    def __init__(self, data_type):
        self.data_type = data_type
        self.tfidf = TfidfVectorizer()

    def transform(self, text):
        self.__load(f'models/tfidf/{self.data_type}/vectorizer.pickle')
        return self.tfidf.transform([text]).toarray()

    def get_feature_names(self):
        return self.tfidf.get_feature_names_out()

    def create_train_data(self, path_data, path_target):
        target = self.__transform_targets(path_target)
        phrase = self.__transform_phrases(path_data)
        
        self.__save_vectorizer(f'models/tfidf/{self.data_type}/vectorizer.pickle')

        phrase = pd.DataFrame(phrase.toarray(), columns = self.get_feature_names())

        target.to_parquet(f'data/train/{self.data_type}/train-target.parquet', index = False, engine = 'pyarrow')
        phrase.to_parquet(f'data/train/{self.data_type}/train-phrase.parquet', index = False, engine = 'pyarrow')

    def __transform_targets(self, path):
        data = self.__read_data(path).iloc[:, 0]
        return pd.get_dummies(data)

    def __transform_phrases(self, path):
        data = self.__read_data(path)
        return self.tfidf.fit_transform(data.phrase_cleaned)

    def __read_data(self, path):
        return pd.read_parquet(path)

    def __save_vectorizer(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.tfidf, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def __load(self, path):
        with open(path, 'rb') as handle:
            self.tfidf = pickle.load(handle)

if __name__ == '__main__':

    print('Preparing examples train data...')
    examples_td = PrepareTrainData('examples')
    examples_td.create_train_data('data/processed/examples/processed-phrase.parquet', 'data/processed/examples/processed-target.parquet')

    print('Preparing sentiments train data...')
    sentiments_td = PrepareTrainData('sentiments')
    sentiments_td.create_train_data('data/processed/sentiments/processed-phrase.parquet', 'data/processed/sentiments/processed-target.parquet')  
