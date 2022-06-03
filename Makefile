clean_text: 
	c:/Users/dello/Desktop/chatbot-text-classifier/env/Scripts/python.exe c:/Users/dello/Desktop/chatbot-text-classifier/src/features/preprocessing.py $(input)

clean_data:
	c:/Users/dello/Desktop/chatbot-text-classifier/env/Scripts/python.exe c:/Users/dello/Desktop/chatbot-text-classifier/src/features/clean_data.py $(input) 

make_features:
	c:/Users/dello/Desktop/chatbot-text-classifier/env/Scripts/python.exe c:/Users/dello/Desktop/chatbot-text-classifier/src/features/make_features.py $(input) 