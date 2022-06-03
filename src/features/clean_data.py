import os
import sys
import pandas as pd 
from preprocessing import Preprocessing
from tqdm import tqdm 
tqdm.pandas()

class CleanPhrases():

    '''
    Class to apply the preprocessing to the data. 
    '''

    def __init__(self):
        
        self.preprocessing = Preprocessing()

        # input and output data paths
        self.raw_path = 'data/raw/'
        self.processed_path = 'data/processed/'

        self.tfidf = TfidfVectorizer()

    def clean(self, data_path):

        ''' 
        This method cleans the phrases in the data. It receives the name of the dataset in data/raw/ and the name of the output file in data/processed/
        with 'processed' tag.
        ''' 

        # count number of processed data to set the correct version in output file name
        output_version_number = len(os.listdir(self.processed_path))

        # read raw data
        data = self.__read_data(self.raw_path + data_path)

        # create a column with the cleaned phrases
        print('Data loaded, cleaning phrases...\n')
        data.insert(data.shape[1], 'phrase_cleaned', data.phrase.progress_apply(self.preprocessing.text_cleaning))

        # save data
        print(f'Phrases cleaned, saving as parquet in {self.processed_path}\n')
        data.to_parquet(f'data/processed/v{output_version_number}-processed.parquet', index = False, engine = 'pyarrow')

        print(f'Completed with success!\n')

    def __read_data(self, data_path):
        return pd.read_csv(data_path, sep = ';')

if __name__ == '__main__':
    print('Starting preprocessing data...\n')
    mkfeatures = CleanPhrases()
    mkfeatures.clean(sys.argv[1])

