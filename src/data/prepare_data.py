import os
import sys
import pandas as pd 
import random
sys.path.append('src/features')
from preprocessing import Preprocessing
from tqdm import tqdm 
tqdm.pandas()

class PrepareData():

    '''
    Class to apply the preprocessing to the data
    '''

    def __init__(self):
        
        self.preprocessing = Preprocessing()

    def prepare(self):

        ''' 
        This method cleans the phrases in the data. It receives the name of the dataset in data/raw/ and the name of the output file in data/processed/
        with 'processed' tag
        ''' 

        # read raw data files
        target = self.__read_data('data/raw/raw-target.parquet')
        phrase = self.__read_data('data/raw/raw-phrase.parquet')

        # concatenate the data to facilitate balancing
        # remove duplicated rows
        data = pd.concat([target, phrase], axis = 1).drop_duplicates()

        # create a column with the cleaned phrases
        data.insert(data.shape[1], 'phrase_cleaned', data.phrase.progress_apply(self.preprocessing.clean))

        # balance data
        data = self.__balance_data(data)

        # split the data into intent and phrase to save different files
        target = data.intent.to_frame()
        phrase = data.phrase_cleaned.to_frame()

        # save data
        target.to_parquet(f'data/processed/processed-target.parquet', index = False, engine = 'pyarrow')
        phrase.to_parquet(f'data/processed/processed-phrase.parquet', index = False, engine = 'pyarrow')

    def __read_data(self, path):
        return pd.read_parquet(path)

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

                    # maybe the transformation can cause a exception, so we try to avoid it 
                    # replicating the same phrase
                    try:
                        sample = intent_data.sample(1)
                        text = sample.phrase_cleaned.values[0]
                        phrase_augmentaded = self.__random_augmentation(text)
                        data = self.__add_row(data, intent, text, phrase_augmentaded)
                    except:
                        data = self.__add_row(data, intent, text, text)

        return data

    def __random_augmentation(self, text):

        '''
        method to apply data augmentation to the phrase with random transformation
        ''' 

        # select one of the transformations, do nothing have a 60% chance
        random_augmentation_tag = random.choices(
            ['nothing', 'add_char', 'remove_char', 'duplicate_word', 'remove_word'],  
            weights = [0.6, 0.1, 0.1, 0.1, 0.2],
            k = 1
        )

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

        '''
        add a random char to the phrase
        E.g. 'hello world' -> 'helloo world!'
        '''

        len_text = len(text)
        random_char = random.choice(list('abcdefghijklmnopqrtuvxwyz?!.-_0123456789'))
        random_index = random.randint(0, len_text)
        text = text[:random_index] + random_char + text[random_index:]

        return text

    def __remove_char(self, text):

        '''
        remove a random char from the phrase
        E.g. 'hello world' -> 'hello wrld'
        '''

        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index + 1:]
        return text
    
    def __duplicate_word(self, text):

        '''
        take a random word from the phrase and duplicate it
        E.g. 'hello world' -> 'hello hello world'
        '''

        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index] + text[random_index:]
        return text

    def __remove_word(self, text):

        '''
        take a random word from the phrase and remove it
        E.g. 'hello world' -> 'hello'
        '''

        len_text = len(text)
        random_index = random.randint(0, len_text)
        text = text[:random_index] + text[random_index + 1:]
        return text

    def __add_row(self, data, intent, phrase, phrase_augmentaded):

        '''
        add a new row to the dataframe in format [intent, phrase, phrase_augmentaded]
        '''

        data = pd.concat([data, pd.DataFrame([[intent, phrase, phrase_augmentaded]], columns = ['intent', 'phrase', 'phrase_cleaned'])], ignore_index = True, axis = 0)
        return data

if __name__ == '__main__':
    print('Starting preprocessing data...\n')
    mkfeatures = PrepareData()
    mkfeatures.prepare()

