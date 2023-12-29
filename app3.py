# App version 3 --> refaire tout l'entrainement
from flask import Flask, jsonify, request
import pandas as pd  
import mlflow


app = Flask(__name__)

################################################ Nettoyage du texte
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
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

def load_df():
    ct = 'QueryResults (1).csv'
    df = pd.read_csv(ct)
    df['Cleaned_Body'] = df['Body'].apply(transform_bow_fct)
    return df

df = load_df()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

def load_vec(data):
    tfidf_array = tfidf_vectorizer.fit_transform(data['Cleaned_Body']).toarray()
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)
    # Concaténer les DataFrames sans utiliser de sample
    data.reset_index(drop=True, inplace=True)
    return tfidf_df

tfidf_df = load_vec(df)

from sklearn.decomposition import LatentDirichletAllocation

def load_unspv():

    num_topics = 50
    lda_model_sklearn = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model_sklearn.fit(tfidf_df)
    return lda_model_sklearn 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import random as rd

def load_spv():
    X_train, X_test, y_train, y_test = train_test_split(tfidf_df, df['Tags'], test_size=0.2, random_state=42)
    y_train = [y[rd.randint(0, 2)] for y in y_train]
    y_test = [y[rd.randint(0, 2)] for y in y_test]
    naive_bayes = MultinomialNB(fit_prior=True)
    naive_bayes.fit(X_train, y_train)
    return naive_bayes 

lda_model = load_unspv()
supervised = load_spv()

from collections import defaultdict
def load_dico():
    tag_counts = defaultdict(lambda: defaultdict(int))
    # Parcours des lignes pour compter les correspondances entre les tags et les numéros
    for index, row in df.iterrows():
        tags = row['Tags']
        lda_topics = row['lda_predict_sklearn']

        # Comptage des correspondances entre les tags et les numéros
        for topic in lda_topics:
            for tag in tags:
                tag_counts[topic][tag] += 1

    # Convertion du dictionnaire en DataFrame
    results = pd.DataFrame(tag_counts).T
    results.fillna(0, inplace=True)
    results = results.rename_axis('LDA Topic').reset_index()

    topic_to_tag = {}
    tag_to_topic = {}

    # Parcourir les lignes du DataFrame et associer chaque sujet au tag le plus fréquent
    for _, row in results.iterrows():
        topic = row['LDA Topic']
        tags_except_lda = row.drop('LDA Topic')
        
        # Vérifier si le tag a déjà été associé à un autre sujet
        if topic not in topic_to_tag and not tags_except_lda.empty:
            # Si le tag n'a pas encore été associé à un sujet
            most_common_tag = tags_except_lda.idxmax()
            
            # Vérifier si le tag a déjà été associé à un autre sujet
            while most_common_tag in tag_to_topic:
                # Si le tag a déjà été associé, trouver le prochain tag le plus fréquent
                tags_except_lda = tags_except_lda.drop(most_common_tag)
                if tags_except_lda.empty:
                    # Si tous les tags ont déjà été associés, sortir de la boucle
                    break
                most_common_tag = tags_except_lda.idxmax()
            
            # Associer le sujet au tag le plus fréquent
            topic_to_tag[topic] = most_common_tag
            tag_to_topic[most_common_tag] = topic

    return topic_to_tag

topic_to_tag = load_dico()

def encoding(X):
    X_pred_tfidf = tfidf_vectorizer.transform(X)
    return X_pred_tfidf

################################################ Prédiction


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

if __name__ == '__main__':
    #app.run(host="0.0.0.0",port=8081)
    app.run(debug=True,port=8081)
