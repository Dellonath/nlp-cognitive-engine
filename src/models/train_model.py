import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

# loading data from train directory
TRAIN = pd.read_parquet('data/train/train.parquet')
TARGET = pd.read_parquet('data/train/target.parquet')

# model parameters
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = TRAIN.shape[1]))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(TARGET.shape[1], activation='softmax'))

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc']
)

model.fit(TRAIN, TARGET, epochs = 25)

# save the final model in the models directory
model.save('models/keras')