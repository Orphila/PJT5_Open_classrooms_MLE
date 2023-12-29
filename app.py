from flask import Flask, jsonify, request
import pandas as pd  
import mlflow


app = Flask(__name__)

################################################ Nettoyage du texte
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

from bs4 import BeautifulSoup

# Nettoyage
def strip_html_bs(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def tokenizer_fct(sentence):
    """Division de mots en texte + suppression de certains caractères"""
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')','<','>','~']

def stop_word_filter_fct(list_words):
    """Suppression de mots sans information+ ponctuations"""
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def lower_start_fct(list_words):
    """Conversion en lettres minuscules et suppression de préfixes indésirables"""
    lw = [w.lower() for w in list_words if (not w.startswith("@")) and (not w.startswith("#")) and (not w.startswith("http"))]
    return lw

def lemma_fct(list_words):
    """lemmatisation"""
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

def transform_bow_fct(desc_text):
    """fonction de transformation"""
    text_stripped = strip_html_bs(desc_text)
    word_tokens = tokenizer_fct(text_stripped)
    lw = lower_start_fct(word_tokens)
    sw = stop_word_filter_fct(lw)
    transf_desc_text = ' '.join(sw)
    return transf_desc_text

def preprocess(df):

    df['text'] = df['text'].apply(transform_bow_fct)

    return df['text']

def preprocess_text(text):
    cleaned_text = transform_bow_fct(text)
    return cleaned_text

################################################ Encoding TF - IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

def load_vec():
    path = "file:///Users/orphila_adjovi/PJT5_Open_classrooms_MLE/mlruns/0/d315ee800c374d9895df41d5849fd4b6/artifacts/tfidf_vectorizer"
    vec = mlflow.sklearn.load_model(path)
    return vec

vec_tfidf = load_vec()

import json 
import os

def load_dico():
    run_id = "bea21720d6e24655a835b18200e10cb6"
    path = os.path.join("mlruns", "0", run_id, "artifacts", "topic_to_tag.json")
   
    with open(path, 'r') as file:
        doc = json.load(file)

    return doc

topic_to_tag = load_dico()

def encoding(X):
    X_pred_tfidf = vec_tfidf.transform(X)
    return X_pred_tfidf

def load_models():
    model_path_unsupervised = "file:///Users/orphila_adjovi/PJT5_Open_classrooms_MLE/mlruns/0/97bca3a8757940ff990e6833bd3a3672/artifacts/model"  
    model_path_supervised = "file:///Users/orphila_adjovi/PJT5_Open_classrooms_MLE/mlruns/0/dc70cf8a83b04686a8149fd4401421ad/artifacts/model"
    path_w2vc = "file:///Users/orphila_adjovi/PJT5_Open_classrooms_MLE/mlruns/0/1202c1af41b94bcb976ee0b4b18dcc47/artifacts/model"
    unsupervised = mlflow.sklearn.load_model(model_path_unsupervised)
    supervised = mlflow.sklearn.load_model(model_path_supervised)
    wvc = mlflow.sklearn.load_model(path_w2vc)
    return [unsupervised,supervised,wvc]

lda_model = load_models()[0]
supervised = load_models()[1]
wvc = load_models()[2]
################################################ Prédiction
from collections import defaultdict

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'tags': 'sql','pred':'java'})

def predict(text):
    X_pred = pd.DataFrame({'text': [text]})
    # Prétraitement
    X_pred['text'] = X_pred['text'].apply(preprocess_text)
    # Encoding
    X_pred_tfidf = encoding(X_pred['text'])
    #Prédiction supervisée (tfidf)
    predicted_tags_supervised = supervised.predict(X_pred_tfidf).tolist()

    top_n = 3
    top_topics = lda_model.transform(X_pred_tfidf).argsort(axis=1)[:, -top_n:]
    result_numbers = [topics.tolist() for topics in top_topics]
    result_numbers  = [str(float(num)) for num in result_numbers[0]]
    result_tags = [topic_to_tag.get(num, "Tag_inconnu") for num in result_numbers]
    #return topic_to_tag
    return jsonify({'results_supervised': predicted_tags_supervised
                    ,'results_unsupervised_0' :result_tags
                    })
    

################################################ Affichage
@app.route('/predict', methods=['GET'])
def predict_endpoint():
    text = request.args.get('text', '')
    result = predict(text)
    return result
################################################ ??
@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.json
    return jsonify(data)

if __name__ == '__main__':
    #app.run(host="0.0.0.0",port=8081)
    app.run(debug=True,port=8081)
