import os
import sys
import pandas as pd 
from preprocessing import Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

class MakeFeatures():

    def __init__(self):

        self.tfidf = TfidfVectorizer()

        # input and output data paths
        self.processed_path = 'data/processed/'
        self.train_path = 'data/train/'

    def tranform(self, text):
        return self.tfidf.transform([text]).toarray()

    def get_feature_names(self):
        return self.tfidf.get_feature_names_out()

    def make(self, data_path):

        output_version_number = len(os.listdir(self.train_path))

        data = self.__fit_tfidf(self.processed_path + data_path)
        data = pd.DataFrame(data.toarray(), columns = self.get_feature_names())

        data.to_parquet(f'data/train/v{output_version_number}-train.parquet', index = False, engine = 'pyarrow')
    
    def __fit_tfidf(self, data_path):
        data = self.__read_data(data_path)
        return self.tfidf.fit_transform(data.phrase_cleaned)
    
    def __read_data(self, data_path):
        return pd.read_parquet(data_path)

if __name__ == '__main__':
    mkfeatures = MakeFeatures()
    mkfeatures.make(sys.argv[1])