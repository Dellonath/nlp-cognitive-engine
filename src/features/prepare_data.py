import os
import sys
import pandas as pd 
from preprocessing import Preprocessing
import random
from tqdm import tqdm 
tqdm.pandas()

class PrepareData():

    '''
    Class to apply the preprocessing to the data. 
    '''

    def __init__(self):
        
        self.preprocessing = Preprocessing()

    def prepare(self, data_path):

        ''' 
        This method cleans the phrases in the data. It receives the name of the dataset in data/raw/ and the name of the output file in data/processed/
        with 'processed' tag
        ''' 

        # read raw data and drop duplicates
        data = self.__read_data('data/raw/' + data_path).drop_duplicates()

        # create a column with the cleaned phrases
        print('Data loaded, cleaning phrases...\n')
        data.insert(data.shape[1], 'phrase_cleaned', data.phrase.progress_apply(self.preprocessing.text_cleaning))

        # save data
        print(f'\nPhrases cleaned, saving as parquet in data/processed\n')
        data.to_parquet(f'data/processed/processed.parquet', index = False, engine = 'pyarrow')

        print(f'Completed with success!\n')

    def __read_data(self, data_path):
        return pd.read_parquet(data_path)

    def __balance_data(self, data):

        # get the number of the major intent
        major_intent = data.intent.value_counts().max()

        # for each intent, balance the data by oversampling using data augmentation
        for intent in data.intent.unique():
            intent_data = data[data.intent == intent]
            total_augmentation = major_intent - intent_data.shape[0]

            # if the number of the intent is less than the number of the major intent, oversample the data
            if total_augmentation > 0:
                for _ in range(total_augmentation):

                    try:
                        sample = intent_data.sample(1)
                        text = sample.phrase.values[0]

                        phrase_augmentaded = self.__data_augmentation(text)

                        data = pd.concat([data, pd.DataFrame([[intent, phrase_augmentaded]], columns = ['intent', 'phrase'])], ignore_index = True, axis = 0)
                    
                    except:
                        data = pd.concat([data, pd.DataFrame([[intent, text]], columns = ['intent', 'phrase'])], ignore_index = True, axis = 0)

        return data

    def __data_augmentation(self, text):

        '''
        method to apply data augmentation to the phrase with random transformation
        ''' 

        random_augmentation_tag = random.choice(['nothing', 'add_char', 'remove_char', 'duplicate_word', 'remove_word'], p = [0.5, 0.125, 0.125, 0.125, 0.125])

        if random_augmentation_tag == 'add_char':
            text = self.__add_char(text)
        elif random_augmentation_tag == 'remove_char':
            text = self.__remove_char(text)
        elif random_augmentation_tag == 'duplicate_word':
            text = self.__duplicate_word(text)
        elif random_augmentation_tag == 'remove_word':
            text = self.__remove_word(text)
        else:
            pass
        
        return text

    def __add_char(self, text):

        len_text = len(text)
        random_char = random.choice(list('abcdefghijklmnopqrtuvxwyz?!.-_0123456789'))
        random_index = random.randint(0, len_text)
        text = text[:random_index] + random_char + text[random_index:]

        return text

    def __remove_char(self, text):

        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index + 1:]
        return text
    
    def __duplicate_word(self, text):

        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index] + text[random_index:]
        return text

    def __remove_word(self, text):
        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index + 1:]
        return text

if __name__ == '__main__':
    print('Starting preprocessing data...\n')
    mkfeatures = PrepareData()
    mkfeatures.prepare(sys.argv[1])

