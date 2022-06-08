
# organize the data
prepare_data:
	env/bin/python3 src/data/prepare_data.py

# generate the train dataset
train_data:
	env/bin/python3 src/data/train_data.py 

prepare: prepare_data train_data

# fit the examples model
train_examples:
	env/bin/python3 src/models/examples_model.py

# fit the sentiments model
train_sentiments:
	env/bin/python3 src/models/sentiments_model.py

train: train_examples train_sentiments

# predict infos of a message
predict:
	env/bin/python3 src/models/predict_model.py $(input)

# prepare the project to be used
all: prepare train

# host the models API
deploy:
	env/bin/python3 src/api/api.py    

# remove __pycache__ and .ipynb_checkpoints folders if exists
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 