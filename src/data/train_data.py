import os
import sys
import pandas as pd 
import pickle
sys.path.append('src/features')
from preprocessing import Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

class PrepareTrainData():

    def __init__(self):
        self.tfidf = TfidfVectorizer()

    def transform(self, text):
        self.__load()
        return self.tfidf.transform([text]).toarray()

    def get_feature_names(self):
        return self.tfidf.get_feature_names_out()

    def create_train_data(self):
        target = self.__transform_targets()
        phrase = self.__transform_phrases()
        
        self.__save_vectorizer()

        phrase = pd.DataFrame(phrase.toarray(), columns = self.get_feature_names())

        target.to_parquet(f'data/train/train-target.parquet', index = False, engine = 'pyarrow')
        phrase.to_parquet(f'data/train/train-phrase.parquet', index = False, engine = 'pyarrow')

    def __transform_targets(self, path = 'data/processed/processed-target.parquet'):
        data = self.__read_data(path).iloc[:, 0]
        return pd.get_dummies(data)

    def __transform_phrases(self, path = 'data/processed/processed-phrase.parquet'):
        data = self.__read_data(path)
        return self.tfidf.fit_transform(data.phrase_cleaned)

    def __read_data(self, path):
        return pd.read_parquet(path)

    def __save_vectorizer(self, path = 'models/tfidf/vectorizer.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump(self.tfidf, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def __load(self, path = 'models/tfidf/vectorizer.pickle'):
        with open(path, 'rb') as handle:
            self.tfidf = pickle.load(handle)

if __name__ == '__main__':
    td = PrepareTrainData()
    td.create_train_data()
