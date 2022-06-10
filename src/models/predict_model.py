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
        
        # load the intents and sentiments columns (labels) to preserve the order of the one hot target for each one
        self.intents = pd.read_parquet('data/train/examples/train-target.parquet').columns
        self.sentiments = pd.read_parquet('data/train/sentiments/train-target.parquet').columns

        self.RESPONSES = pd.read_csv('data/external/responses-phrases.csv', sep = ';')

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
        top_five_intents = self.__get_topn_confidences(examples_model_prediction, self.intents, 5)
        top_three_sentiments = self.__get_topn_confidences(sentiments_model_prediction, self.sentiments, 3)

        # the best intent
        best_intent = top_five_intents[0]
        best_intent_name = best_intent['name']
        best_intent_confidence = best_intent['confidence']

        # if the confidence is low, we return the response for the intent
        best_intent_name = best_intent_name if float(best_intent_confidence) > 0.3 else 'irrelevant'

        # get now datetime
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # get the response
        response = self.__get_response(best_intent_name)

        # phrase sentiment
        # top_three_sentiments = [{"negative": "0.0663452819"}, {"neutral": "0.0025401216"}, {"positive": "0.9311146140"}] (not exactly in this order)

        sentiments = {}
        sentiments[top_three_sentiments[0]['name']] = top_three_sentiments[0]['confidence']
        sentiments[top_three_sentiments[1]['name']] = top_three_sentiments[1]['confidence']
        sentiments[top_three_sentiments[2]['name']] = top_three_sentiments[2]['confidence']

        output = {
            'message': {
                'text': text, # original user phrase
                'cleaned': text_cleaned, # phrase that was cleaned and used to predict
                'sentiment': sentiments
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

    def __get_topn_confidences(self, prediction_array, labels, topn = 5):

        # save a list of confidence for each intent
        intents_array = [{'name': labels[i], 'confidence': '{:.10f}'.format(prediction_array[i])} for i in range(len(labels))]

        # organize the output
        # sort the intents by confidence and select top n
        top_five_intents = sorted(intents_array, key = lambda element: element['confidence'], reverse = True)[:topn]

        return top_five_intents

    def __get_response(self, intent):
        return self.RESPONSES.query(f'target == "{intent}"').response.values[0]

if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])
    predictor = Predict()
    print(predictor.predict(text))
