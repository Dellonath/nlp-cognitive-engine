<h1 align="center">
    Cognitive Engine for Natural Language Processing
</h1>

<p align="center">
    <img width=600px src="https://user-images.githubusercontent.com/56659549/171969411-476203ab-d016-4946-a163-ee8b1f6de37d.jpg">
</p>

<h2 align="center">
    A Cognitive Engine using the KERAS framework and other tools to be used as a Natural Language Processing engine in Chatbots
</h2>

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
    How to use
</h2>


First of all, the data <b>must be</b> in the same folder (data/raw) and in the following format (with these columns names and in .parquet format):

| intent  | phrase |
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

Where the intent is the name of the intent and the phrase is the phrase that will be used to train the model. 

In addition, is needed to split the intent (target) column and phrase column into two different files:
*  raw-phrase.parquet for messages column;
*  raw-target.parquet for intent column.

This repository has an example of data in portuguese, but it can be used in any language, you just need to change the raw data files.
_____________

Now, you need clone the repository in your computer.

```terminal
  git clone https://github.com/Dellonath/chatbot-cognitive-engine.git
```

After that, you can run the following command to install all the dependencies:
  
```terminal
  python3 -m venv env 
  source env/bin/activate
  pip install -r requirements.txt
```

At the end, you can run the following command to run the application and prepare it to be used:
```terminal
  make all
```

This command will prepare the data, train the model and save it in the model folder. After that, you can run the following command to predict intent to a message:

```terminal
  make predict input="I want a credit card"
```

The output must be a json with some infos about the message and the predicted intent:

```json
{
    "user": {
        "text": "I want a credit card", 
        "cleaned": "i want a credit card"
    }, 
    "intent": {
        "name": "credit_card", 
        "confidence": "0.9811101042"
    }, 
    "intents": [
        {"intent": "credit_card", "confidence": "0.9811101042"}, 
        {"intent": "create_account", "confidence": "0.0079928385"}, 
        {"intent": "change_password", "confidence": "0.0048914794"}, 
        {"intent": "alter_personal_data", "confidence": "0.0019387764"}, 
        {"intent": "check_credit_limit", "confidence": "0.00084826007"}
    ]
}
```

Where the user is the message that will be predicted, the intent is the predicted intent and the intents is a list of top 5 intents with their confidence.

<h2 align="center">
    API
</h2>

You can execute the Cognitive Engine as a API to predict an intent of a message:

```terminal
make deploy
```

Now, just access the following url in your browser:

```url
http://127.0.0.1:5000/predict?text=<TEXT>
```

E.g.
```
http://127.0.0.1:5000/predict?text="I want a credit card"
```

<p align="center">
    <img width=700px src="https://user-images.githubusercontent.com/56659549/172020552-bd4c1be7-2608-4058-8388-8829769a834b.png">
</p>

_____________
 
<h3 align="center">
    Project Organization
</h3>

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
       ├── data            <- Scripts to download or generate data
       │   ├── prepare_data.py
       │   └── train_data.py
       │
       ├── features       <- Scripts to turn clean data and prepare to encoder
       │   └── preprocessing.py
       │
       └── models         <- Scripts to train models and then use trained models to make
           │                 predictions
           ├── predict_model.py
           └── train_model.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
