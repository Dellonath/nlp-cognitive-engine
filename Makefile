
prepare_data:
	env/bin/python3 src/data/prepare_data.py

train_data:
	env/bin/python3 src/data/train_data.py 

train:
	env/bin/python3 src/models/train_model.py

predict:
	env/bin/python3 src/models/predict_model.py $(input)

clean: 
	env/bin/python3 src/features/preprocessing.py $(input)

all: prepare_data train_data train
	echo 'Model ready to use!'

run deploy:
	env/bin/python3 src/api/api.py