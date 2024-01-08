# App version 2 --> charger les fichiers modèles
from flask import Flask, jsonify, request
import pandas as pd  

app = Flask(__name__)

################################################ Nettoyage du texte
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from bs4 import BeautifulSoup

# Nettoyage
def strip_html_bs(text):
    """nettoyage des balises html"""
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
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem = lemma_fct(lw) 
    transf_desc_text = ' '.join(lem)
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
import json
import pickle

def load_vec():
    path = "vec.pkl"
    with open(path, 'rb') as file:
        vc = pickle.load(file)
    return vc

tfidf_vectorizer = load_vec()

def load_dico():
    path =  "topic_to_tag_2.json"
    with open(path, 'r') as file:
        doc = json.load(file)
    return doc

topic_to_tag = load_dico()

def encoding(X):
    X_pred_tfidf = tfidf_vectorizer.transform(X)
    return X_pred_tfidf

def load_mlb():
    path = "mlb.pkl"
    with open(path, 'rb') as file:
        mb = pickle.load(file)
    return mb

mlb = load_mlb()

def load_models():
    model_path_unsupervised = "LDA.pkl"  
    model_path_supervised = "spv.pkl"
    with open(model_path_unsupervised, 'rb') as file:
        doc1 = pickle.load(file)
    with open(model_path_supervised, 'rb') as file:
        doc2 = pickle.load(file)
    return [doc1,doc2]

lda_model = load_models()[0]
supervised = load_models()[1]

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
    predicted_tags_nb_bin = supervised.predict(X_pred_tfidf)
    predicted_tags_list = [list(tags) if tags else ['no_result'] for tags in mlb.inverse_transform(predicted_tags_nb_bin)]

    top_topics = lda_model.transform(X_pred_tfidf).argsort(axis=1)[:, ::-1]
    result_numbers = [topics.tolist() for topics in top_topics][0][:5]
    result_tags = [topic_to_tag.get(str(float(num)), "Tag_inconnu") for num in result_numbers]

    return jsonify({'results_supervised': predicted_tags_list
                    ,'results_unsupervised' :result_tags
                    })
    

################################################ Affichage
@app.route('/predict', methods=['GET'])
def predict_endpoint():
    text = request.args.get('text', '')
    result = predict(text)
    return result
################################################ Launch

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=8581)

