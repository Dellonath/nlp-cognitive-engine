import sys
import unidecode
import re

class Preprocessing():

    def __init__(self):
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stopwords = self.__load_stopwords()
    
    def clean(self, text):
        # text = self.__remove_stopwords(text)
        text = self.__remove_tweets_tags(text)
        #text = self.__remove_punctuation(text)
        #text = self.__remove_accents(text)
        text = self.__remove_numbers(text)
        text = self.__lower_case(text)
        
        return text
    
    def __remove_accents(self, text):
        return unidecode.unidecode(text)

    def __remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def __remove_stopwords(self, text):
        return ' '.join([word for word in text.split() if word not in self.stopwords])

    def __remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def __lower_case(self, text):
        return text.lower()
    
    def __load_stopwords(self):
        return #open('data/external/stopwords.csv', 'r', encoding = 'utf-8').read().split('\n')

    def __remove_tweets_tags(self, text):
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'https?:\/\/\S+', '', text)
            text = re.sub(r'&\w+', '', text)
            text = re.sub(r'RT', '', text)
            return text

if __name__ == '__main__':
    preprocess = Preprocessing()
    print(preprocess.clean(' '.join(sys.argv[1:]))) 