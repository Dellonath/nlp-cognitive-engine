import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
class CognitiveEngine():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model = Sequential([
            Input(self.input_shape),
            Dense(units = 128, activation = 'relu'),
            Dense(units = 64, activation = 'relu'),
            Dense(self.output_shape, activation = 'softmax')
        ]) 

        self.model.compile(
            optimizer = 'adam', 
            loss = 'categorical_crossentropy', 
            metrics = ['acc']
        )

    def train(self, train_data, train_labels, epochs = 25):

        '''
        method to train the model
        '''

        self.model.fit(train_data, train_labels, epochs = epochs)

    def save_model(self, path = 'models/keras/model'):
        # save the final model in the models directory
        self.model.save(path, save_format = 'h5')

if __name__ == '__main__':

    # loading data from train directory
    TRAIN = pd.read_parquet('data/train/train-phrase.parquet').sample(frac=1)
    TARGET = pd.read_parquet('data/train/train-target.parquet').iloc[TRAIN.index]

    # get the shape of the data
    input_shape = TRAIN.shape[1]
    output_shape = TARGET.shape[1]

    # create, fit and save the model
    CE = CognitiveEngine(input_shape, output_shape)
    CE.train(TRAIN, TARGET)
    CE.save_model()


