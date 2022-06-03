clean: 
	env/Scripts/python.exe src/features/clean_text.py $(input)

prepare_data:
	env/Scripts/python.exe src/features/prepare_data.py $(input) 

train_data:
	env/Scripts/python.exe src/features/train_data.py $(input) 

train:
	env/Scripts/python.exe src/models/train_model.py

predict:
	env/Scripts/python.exe src/models/predict_model.py $(input)