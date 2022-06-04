
prepare_data:
	env/Scripts/python.exe src/features/prepare_data.py

train_data:
	env/Scripts/python.exe src/features/train_data.py 

train:
	env/Scripts/python.exe src/models/train_model.py

predict:
	env/Scripts/python.exe src/models/predict_model.py $(input)

clean: 
	env/Scripts/python.exe src/features/preprocessing.py $(input)

all: prepare_data train_data train
	echo 'Model ready to use!'