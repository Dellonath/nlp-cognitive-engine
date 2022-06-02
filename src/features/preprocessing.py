
import unidecode

class Preprocessing():

    def __init__(self):
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stopwords = __load_stopwords()
    
    def text_cleaning(self, text):
        pass 
    
    def __remove_accents(self, text):
        return unidecode.unidecode(text)

    def __remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', self.punctuation))

    def __remove_stopwords(self, text):
        return ' '.join([word for word in text.split() if word not in self.stopwords])
    
    def __load_stopwords(self):
        return open('../../data/external/stopwords.txt', 'r').read().split('\\n')
