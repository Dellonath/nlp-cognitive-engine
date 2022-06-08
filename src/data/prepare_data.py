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

    def __init__(self, data_type):
        self.data_type = data_type
        self.preprocessing = Preprocessing()
        
    def prepare(self, path):

        ''' 
        This method cleans the phrases in the data. It receives the name of the dataset in data/raw/ and the name of the output file in data/processed/
        with 'processed' tag
        ''' 

        # read raw data
        data = self.__read_data(path).drop_duplicates(['phrase'])

        # create a column with the cleaned phrases
        data.insert(data.shape[1], 'phrase_cleaned', data.phrase.progress_apply(self.preprocessing.clean))

        # balance data
        data = self.__balance_data(data)

        # split the data into target and phrase to save different files
        target = data.target.to_frame() # target
        phrase = data.phrase_cleaned.to_frame()

        # save data
        target.to_parquet(f'data/processed/{self.data_type}/processed-target.parquet', index = False, engine = 'pyarrow')
        phrase.to_parquet(f'data/processed/{self.data_type}/processed-phrase.parquet', index = False, engine = 'pyarrow')

    def __read_data(self, path):
        return pd.read_parquet(path)

    def __balance_data(self, data, target_column_name = 'target'):

        # get the number of the major target class
        major_class = data[target_column_name].value_counts().max()

        # for each target class, balance the data by oversampling using data augmentation
        for target in data[target_column_name].unique():
            class_data = data[data[target_column_name] == target]
            total_augmentation = major_class - class_data.shape[0]

            # if the number of the target is less than the number of the major target, oversample the data
            if total_augmentation > 0:
                for _ in range(total_augmentation):

                    # maybe the transformation can cause a exception, so we try to avoid it 
                    # replicating the same phrase
                    sample = class_data.sample(1)
                    text = sample.phrase_cleaned.values[0]
                    try:
                        text_augmentaded = self.__random_augmentation(text)
                        data = self.__add_row(data, target, text, text_augmentaded)
                    except:
                        data = self.__add_row(data, target, text, text)

        return data

    def __random_augmentation(self, text):

        '''
        method to apply data augmentation to the phrase with random transformation
        ''' 

        # select one of the transformations, do nothing have a 60% chance
        random_augmentation_tag = random.choices(
            ['nothing', 'add_char', 'remove_char', 'duplicate_word', 'remove_word'],  
            weights = [0.5, 0.125, 0.125, 0.125, 0.125],
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

    def __add_row(self, data, target, phrase, phrase_augmentaded):

        '''
        add a new row to the dataframe in format [target, phrase, phrase_augmentaded]
        '''

        data = pd.concat([data, pd.DataFrame([[target, phrase, phrase_augmentaded]], columns = ['target', 'phrase', 'phrase_cleaned'])], ignore_index = True, axis = 0)
        return data

if __name__ == '__main__':

    examples_mkfeatures = PrepareData('examples')
    examples_mkfeatures.prepare('data/raw/examples/raw.parquet')

    sentiments_mkfeatures = PrepareData('sentiments')
    sentiments_mkfeatures.prepare('data/raw/sentiments/raw.parquet')

