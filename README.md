<h1 align="center">
  Cognitive Engine for Natural Language Processing
</h1>

<p align="center">
  <img width=500px src="https://user-images.githubusercontent.com/56659549/171969411-476203ab-d016-4946-a163-ee8b1f6de37d.jpg">
</p>

<h3 align="center">
  A Cognitive Engine using the KERAS Deep Learning Framework to be used as a Natural Language Processing engine in Chatbots
</h3>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/progress-100%25-important.svg?color=greeb&style=for-the-badge">
  <a href="https://github.com/Dellonath/SKADI/blob/main/LICENSE">
    <img src="https://img.shields.io/apm/l/vim-mode?color=greeb&style=for-the-badge">
  </a>
  <a href="https://github.com/Dellonath/chatbot-cognitive-engine/stargazers">
    <img src="https://img.shields.io/github/stars/Dellonath/chatbot-cognitive-engine?color=greeb&style=for-the-badge">
  </a> 
  <a href="https://github.com/Dellonath/chatbot-cognitive-engine/network/members">
    <img src="https://img.shields.io/github/forks/Dellonath/chatbot-cognitive-engine?color=greeb&style=for-the-badge">
  </a>
</p>

<br>
 
<h2 align="center">
  Considerations
</h2>

First of all, the data file <b>must be</b> in the following format (with these columns names and in ```.parquet``` format):

| target  | phrase |
| ------------- | ------------- |
| credit_card  | I would like a credit card |
| credit_card  | How I can get a credit card? |
| credit_card  | I wish for a credit card? |
| ...  | ...  |
| change_password  | I want to change my password |
| change_password  | Can I change my password on the app? |
| ...  | ...  |
| create_account  | Do I need a credit card to create a account? |
| create_account  | I would like to create a account in this bank |
| create_account  | How do I create an account in this bank? |

Where the ```target``` is the name of the target (intent) and ```phrase``` is the example message. In addition, the file name <b>must be</b> ```raw.parquet``` for both data/raw/examples.
<b>This dataset will be used to train de Cognitive Engine</b>

Another file is needed: the responses. In data/external is necessary to have a file with the ```responses-phrases.csv``` (with this name). This file must have the following columns:
*  target: the name of the intent;
*  response: the response that will be used as a response to the intent.

| target  | response |
| ------------- | ------------- |
| credit_card  | I guess you want a credit card. You can create an account in this bank and then you can receive your credit card. Thank you. |
| alter_personal_data  | You can edit your personal data in your Profile on our App. Take a look!  |
| change_password  | Do you need to change your password? You can do it in our app. |
| ...  | ...  |
| create_account  | Welcome to our bank. We need some personal information to create your account. Feel free with your money :). |

<h2 align="center">
  Installation
</h2>

You need clone the repository in your computer:
```terminal
  git clone https://github.com/Dellonath/nlp-cognitive-engine.git
```

This repository has an example of data in Portuguese, but it can be used in any language, you just need to change the raw data files.

After that, you can run the following command to create a virtual environment and install all the dependencies:  
```terminal
  python3 -m venv env 
  source env/bin/activate
  pip install -r requirements.txt
```

Then, you can run the following command to run the application and prepare it to be used. This command will prepare the data, train the models and save it in the model folder:
```terminal
  make all
```

At least, you can execute the Cognitive Engine as an API to predict the intent of a message:

```terminal
make deploy
```

Now, just access the following URL in your browser:

```url
http://127.0.0.1:5000/predict?text=<user-message>
```

E.g.:
```url
http://127.0.0.1:5000/predict?text="I want a credit card"
```

The output must be a json with some infos about the message and the predicted intent:

```json
{
    "message": {
        "text": "I want a credit card", 
        "cleaned": "i want a credit card",
        "sentiment": {
            "positive": "0.0015173794",
            "neutral": "0.9981304513",
            "negative": "0.0003521693",
        }
    }, 
    "intent": {
        "name": "credit_card", 
        "confidence": "0.9811101042"
    }, 
    "intents": [
        {"name": "credit_card", "confidence": "0.9811101042"}, 
        {"name": "create_account", "confidence": "0.0079928385"}, 
        {"name": "change_password", "confidence": "0.0048914794"}, 
        {"name": "alter_personal_data", "confidence": "0.0019387764"}, 
        {"name": "check_credit_limit", "confidence": "0.00084826007"}
    ],
    "response": "I guess you want a credit card. You can create an account in this bank and then you can receive your credit card. Thank you.",
    "created_at": "2022-06-07 18:58:11",
}
```

The ```message``` have three fields: ```text```, ```cleaned``` and ```sentiment```. The ```text``` is the message without processing. The ```cleaned``` is the message that will be used to predict the intent. The ```sentiment``` is the user sentiment when wrote this message. The field ```intent``` have the most relevant intent (major confidence). The ```intents``` have a list of top five intents by confidence. The field ```response``` is the response that will be used as a response to the user. At least, ```created_at``` is the date and time when the message was predicted.
 
<h2 align="center">
  Project Organization
</h2>

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
       │
       ├── api             <- Scripts to make an API.
       │   └── api.py      <- Script to serve the model as an API.
       │
       ├── data            <- Scripts to download or generate data.
       │   ├── prepare_data.py
       │   └── train_data.py
       │
       ├── features        <- Scripts to turn clean data and prepare to encoder.
       │   └── preprocessing.py
       │
       └── models          <- Scripts to train models and then use trained models to make
           │                  predictions.
           ├── predict_model.py
           ├── examples_model.py
           └── sentiment_model.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
