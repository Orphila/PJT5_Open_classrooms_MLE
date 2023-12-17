from flask import Flask, jsonify, request
import pandas as pd  
import mlflow
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

################################################ Importation du dataset
#@app.before_first_request
"""
def load():
    ct = 'QueryResults (1).csv'
    df = pd.read_csv(ct)
    return df

df = load()
"""
################################################ Nettoyage du texte
import pickle
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('word_tokenize')

custom_punkt_path = '/Users/orphila_adjovi/PJT5_Open_classrooms_MLE/corpora/punkt'
punkt_path = nltk.data.find(f'{custom_punkt_path}/english.pickle')

with open(punkt_path, 'rb') as file:
    punkt_model = pickle.load(file)
nltk.data.path.append(custom_punkt_path)

with open('/Users/orphila_adjovi/PJT5_Open_classrooms_MLE/corpora/stopwords/english', 'r') as file:
        custom_stopwords = file.read().splitlines()

# Nettoyage
def tokenizer_fct(sentence):
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    #word_tokens = word_tokenize(sentence_clean, language='english', path_to_punkt=punkt_path)
    word_tokens = punkt_model.tokenize(sentence_clean)
    return word_tokens


stop_w = custom_stopwords + ['[', ']', ',', '.', ':', '?', '(', ')','<','>','~']
def stop_word_filter_fct(list_words):
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def lower_start_fct(list_words):
    lw = [w.lower() for w in list_words if (not w.startswith("@")) and (not w.startswith("#")) and (not w.startswith("http"))]
    return lw

def lemma_fct(list_words):
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

def transform_bow_fct(desc_text):
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    transf_desc_text = ' '.join(lw)
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
    model_path_supervised = "file:///Users/orphila_adjovi/PJT5_Open_classrooms_MLE/mlruns/0/d315ee800c374d9895df41d5849fd4b6/artifacts/model"
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
