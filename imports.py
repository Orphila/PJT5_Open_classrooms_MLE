#### Répétition des fichiers 3 et 4 au format .py pour permettre des tests ####


import pandas as pd
ct = 'QueryResults (1).csv'
df = pd.read_csv(ct)

############################################# Filtre des tags #############################################

df['Tags'] = df['Tags'].str.replace('><', ',').str.replace('<', '').str.replace('>', '')

from collections import Counter
# Fréquence des mots dans 'Tags'
all_tags = df['Tags'].str.split(',').explode().tolist()
tag_counts = Counter(all_tags)

# Trouver les 50 mots les plus fréquents
top_50_tags = [tag for tag, _ in tag_counts.most_common(50)]

# Filtrer les listes de mots pour ne contenir que les 50 mots les plus fréquents
def filter_tags(tags):
    return [tag for tag in tags if tag in top_50_tags]

df['Tags'] = df['Tags'].str.split(',').apply(filter_tags)

# Filtrer les lignes où 'Tags' est vide après le filtrage
df = df[df['Tags'].apply(len) > 0]

############################################# Nettoyage du texte #############################################

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

def stemmer_fct(list_words):
    """Stemming"""
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in list_words]
    return stemmed_words

def transform_bow_fct(desc_text):
    """fonction de transformation"""
    text_stripped = strip_html_bs(desc_text)
    word_tokens = tokenizer_fct(text_stripped)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem = stemmer_fct(lw) 
    transf_desc_text = ' '.join(lem)
    return transf_desc_text

def preprocess(df):

    df['text'] = df['text'].apply(transform_bow_fct)

    return df['text']

def preprocess_text(text):
    cleaned_text = transform_bow_fct(text)
    return cleaned_text
df['Cleaned_Body'] = df['Body'].apply(transform_bow_fct)

############################################# Encoding TF - IDF  #############################################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_array = tfidf_vectorizer.fit_transform(df['Cleaned_Body']).toarray()

# Liste des termes correspondant aux colonnes de la matrice
feature_names = tfidf_vectorizer.get_feature_names_out()
# Scores de chaque mot
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

############################################# unsupervised #############################################

import gensim

from sklearn.decomposition import LatentDirichletAllocation
corpus = df['Cleaned_Body']
X = tfidf_df

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
terms = tfidf_vectorizer.get_feature_names_out()

# Convertion de la matrice TF-IDF en une représentation sparse de Gensim
corpus_tfidf_gensim = gensim.matutils.Sparse2Corpus(tfidf_matrix.T)

# Création d'un dictionnaire Gensim à partir de TF-IDF
dictionary = gensim.corpora.Dictionary.from_corpus(corpus_tfidf_gensim, id2word=dict((i, s) for i, s in enumerate(terms)))

# Entraînement du modèle LDA avec Gensim et sklearn
num_topics = 50
lda_model_gensim= gensim.models.LdaModel(corpus=corpus_tfidf_gensim, id2word=dictionary, num_topics=num_topics, random_state=42)
lda_model_sklearn = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model_sklearn.fit(X)

# Convertir les tags en ensembles
set_tags = df['Tags'].apply(set)

# Obtention des n sujets les plus probables pour chaque ligne de tfidf_df, où n est le nombre de tags
top_topics_gensim = lda_model_gensim.get_document_topics(corpus_tfidf_gensim)
top_topics_gensim = [sorted(topics, key=lambda x: x[1], reverse=True)[:len(tags)] for topics, tags in zip(top_topics_gensim,set_tags)]

top_topics_sklearn = lda_model_sklearn.transform(X).argsort(axis=1)[:, ::-1]
top_topics_sklearn = [topics[:len(tags)] for topics, tags in zip(top_topics_sklearn, set_tags)]

# Création d'une colonne 'lda_predict' avec la liste des sujets les plus probables
df['lda_predict_gensim'] = [topics for topics in top_topics_gensim]
df['lda_predict_gensim'] = df['lda_predict_gensim'].apply(lambda x: [topic[0] for topic in x])
df['lda_predict_sklearn'] = [topics for topics in top_topics_sklearn]

############################################# supervised #############################################

import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score


rd.seed(0)

def modelisation(enc,name):
    # Conversion des tags en une représentation binaire
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(df['Tags'])

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(enc, y_bin, test_size=0.2, random_state=42)

    nb_model = MultinomialNB()
    multioutput_nb = MultiOutputClassifier(nb_model)
    multioutput_nb.fit(X_train, y_train_bin)
    # Initialisation et entrainement du modèle


    # Prédictions
    predicted_tags_nb_bin = multioutput_nb.predict(X_test)

    # Calculer le score Jaccard pour le modèle Random Forest
    jaccard_score_nb = jaccard_score(y_test_bin, predicted_tags_nb_bin, average='micro')


    return("Le score supervisé avec ",name," est de ",round(jaccard_score_nb,2))

modelisation(tfidf_df,'tf_idf')
