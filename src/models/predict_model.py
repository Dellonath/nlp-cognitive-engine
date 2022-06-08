from tensorflow import keras
import pandas as pd
import sys
import pickle
import datetime as datetime

sys.path.append('src/features')
from preprocessing import Preprocessing
class Predict():
    
    def __init__(self):
        self.MODEL = keras.models.load_model('models/keras/model.h5')
        self.TFIDF = pickle.load(open('models/tfidf/vectorizer.pickle', 'rb'))
        self.TARGET = pd.read_parquet('data/train/train-target.parquet').columns
        self.RESPONSES = pd.read_parquet('data/external/responses-phrases.parquet')

    def predict(self, text):

        ''' 
        function to return the model predicion for a given text
        '''

        text_cleaned = Preprocessing().clean(text)

        # prepare text for prediction
        text_encoded = self.__transform(text_cleaned)

        # make prediction
        predicted = self.__model_predict(text_encoded)

        # get the top 5 intents
        top_five_intents = self.__get_top_five_intents(predicted)

        # the best intent
        best_intent = top_five_intents[0]
        best_intent_name = best_intent['intent']
        best_intent_confidence = best_intent['confidence']

        # get now datetime
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # get the response
        response = self.__get_response(best_intent_name)

        output = {
            'user': {
                'text': text, # original user phrase
                'cleaned': text_cleaned # phrase that was cleaned and used to predict
            },
            'intent': {
                'name': best_intent_name,
                'confidence': best_intent_confidence
            },            
            'intents': top_five_intents, # list of intents
            'response': response, # response
            'created_at': created_at # datetime of response
        }
        
        return output

    def __transform(self, text):
        return self.TFIDF.transform([text]).toarray()

    def __model_predict(self, text):
        return self.MODEL.predict(text)[0]

    def __get_top_five_intents(self, prediction_array):

        # save a list of confidence for each intent
        intents_array = [{'intent': self.TARGET[i], 'confidence': '{:.10f}'.format(prediction_array[i])} for i in range(len(self.TARGET))]

        # organize the output
        # sort the intents by confidence and select top 5
        top_five_intents = sorted(intents_array, key = lambda element: element['confidence'], reverse = True)[:5]

        return top_five_intents

    def __get_response(self, intent):
        return self.RESPONSES.query(f'intent == "{intent}"').response.values[0]

if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])
    predictor = Predict()
    print(predictor.predict(text))

