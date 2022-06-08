from numpy import positive
from tensorflow import keras
import pandas as pd
import sys
import pickle
import datetime as datetime

sys.path.append('src/features')
from preprocessing import Preprocessing

class Predict():
    
    def __init__(self):
        self.EXAMPLES_MODEL = keras.models.load_model('models/keras/examples/model')
        self.SENTIMENTS_MODEL = keras.models.load_model('models/keras/sentiments/model')

        self.EXAMPLES_TFIDF = pickle.load(open('models/tfidf/examples/vectorizer.pickle', 'rb'))
        self.SENTIMENTS_TFIDF = pickle.load(open('models/tfidf/sentiments/vectorizer.pickle', 'rb'))
        
        self.intent = pd.read_parquet('data/train/examples/train-target.parquet').columns
        self.RESPONSES = pd.read_parquet('data/external/responses-phrases.parquet')

    def predict(self, text):

        ''' 
        function to return the model predicion for a given text
        '''

        text_cleaned = text.lower()
        # text_cleaned = Preprocessing().clean(text)

        # prepare text for prediction
        examples_text_encoded = self.__transform(self.EXAMPLES_TFIDF, text_cleaned)
        sentiments_text_encoded = self.__transform(self.SENTIMENTS_TFIDF, text_cleaned)

        # make prediction
        examples_model_prediction = self.__model_predict(self.EXAMPLES_MODEL, examples_text_encoded)
        sentiments_model_prediction = self.__model_predict(self.SENTIMENTS_MODEL, sentiments_text_encoded)

        # get the top 5 intents
        top_five_intents = self.__get_top_intents(examples_model_prediction, 5)

        # the best intent
        best_intent = top_five_intents[0]
        best_intent_name = best_intent['intent']
        best_intent_confidence = best_intent['confidence']

        # if the confidence is low, we return the response for the intent
        best_intent_name = best_intent_name if float(best_intent_confidence) > 0.3 else 'irrelevant'

        # get now datetime
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # get the response
        response = self.__get_response(best_intent_name)

        # phrase sentiment
        negative = '{:.10f}'.format(sentiments_model_prediction[0])
        neutral = '{:.10f}'.format(sentiments_model_prediction[1])
        positive = '{:.10f}'.format(sentiments_model_prediction[2])
        
        output = {
            'message': {
                'text': text, # original user phrase
                'cleaned': text_cleaned, # phrase that was cleaned and used to predict
                'sentiment': {
                    'positive': positive,
                    'neutral': neutral,
                    'negative': negative        
                }
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

    def __transform(self, vectorizer, text):
        return vectorizer.transform([text]).toarray()

    def __model_predict(self, model, text):
        return model.predict(text)[0]

    def __get_top_intents(self, prediction_array, ntop = 5):

        # save a list of confidence for each intent
        intents_array = [{'intent': self.intent[i], 'confidence': '{:.10f}'.format(prediction_array[i])} for i in range(len(self.intent))]

        # organize the output
        # sort the intents by confidence and select top n
        top_five_intents = sorted(intents_array, key = lambda element: element['confidence'], reverse = True)[:ntop]

        return top_five_intents

    def __get_response(self, intent):
        return self.RESPONSES.query(f'target == "{intent}"').response.values[0]

if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])
    predictor = Predict()
    print(predictor.predict(text))