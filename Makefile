
prepare data:
	env/Scripts/python.exe src/data/prepare_data.py

train data:
	env/Scripts/python.exe src/data/train_data.py 

train:
	env/Scripts/python.exe src/models/train_model.py

predict:
	env/Scripts/python.exe src/models/predict_model.py $(input)

clean: 
	env/Scripts/python.exe src/features/preprocessing.py $(input)

all: prepare_data train_data train
	echo 'Model ready to use!'

run deploy:
	env/Scripts/python.exe src/api/api.py