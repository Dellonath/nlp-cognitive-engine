from tensorflow import keras
import pandas as pd
import sys
import pickle

# load models and target label
MODEL = keras.models.load_model('models/keras')
TFIDF = pickle.load(open('models/tfidf/tfidf_vectorizer.pickle', 'rb'))
TARGET = pd.read_parquet('data/train/target.parquet').columns

def predict(text):

    ''' 
    function to return the model predicion for a given text
    '''

    # prepare text for prediction
    text_encoded = TFIDF.transform([text]).toarray()

    # make prediction
    predicted = MODEL.predict(text_encoded)[0]

    # save a list of confidence for each intent
    intents = [{'intent': TARGET[i], 'confidence': predicted[i]} for i in range(len(predicted))]

    # organize the output
    # sort the intents by confidence and select top 5
    sorted_intents = sorted(intents, key = lambda element: element['confidence'], reverse = True)[:5]

    # the best intent
    best_intent = sorted_intents[0]
    best_intent_name = best_intent['intent']
    best_intent_confidence = best_intent['confidence']

    output = {'phrase': text, 'intent': best_intent_name, 'confidence': best_intent_confidence, 'intents': sorted_intents}
    
    return output

if __name__ == '__main__':
    print(predict(' '.join(sys.argv[1:])))

