import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Input

class SentimentEngine():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model = Sequential([
            Input(self.input_shape),
            Dense(units = 256, activation = 'relu'),
            Dense(units = 128, activation = 'relu'),
            Dense(self.output_shape, activation = 'softmax')
        ]) 

        self.model.compile(
            optimizer = 'adam', 
            loss = 'categorical_crossentropy', 
            metrics = ['acc']
        )

    def train(self, train_data, train_targets, epochs = 10, validation_split = 0.15):

        '''
        method to train the model
        '''

        self.model.fit(
            train_data, 
            train_targets, 
            epochs = epochs,
            validation_split = validation_split
        )

    def save_model(self, path):
        # save the final model in the models directory
        self.model.save(path, save_format = 'h5')

if __name__ == '__main__':

    # loading data from train directory
    TRAIN = pd.read_parquet('data/train/sentiments/train-phrase.parquet').sample(frac = 1)
    TARGET = pd.read_parquet('data/train/sentiments/train-target.parquet').iloc[TRAIN.index]

    # get the shape of the data
    input_shape = TRAIN.shape[1]
    output_shape = TARGET.shape[1]

    # create, fit and save the model
    SE = SentimentEngine(input_shape, output_shape)
    SE.train(TRAIN, TARGET)
    SE.save_model('models/keras/sentiments/model')
