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

df['nb_tags'] = df['Tags'].apply(lambda x: len(x))
df = df[df['nb_tags']==3].reset_index(drop=True)

############################################# Nettoyage du texte #############################################

# @title Nettoyage du texte
import pickle
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

custom_punkt_path = '/Users/orphila_adjovi/PJT5_Open_classrooms_MLE/corpora/punkt'
punkt_path = nltk.data.find(f'{custom_punkt_path}/english.pickle')

with open(punkt_path, 'rb') as file:
    punkt_model = pickle.load(file)
nltk.data.path.append(custom_punkt_path)

with open('/Users/orphila_adjovi/PJT5_Open_classrooms_MLE/corpora/stopwords/english', 'r') as file:
    custom_stopwords = file.read().splitlines()
    
# Nettoyage
def strip_html_bs(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def tokenizer_fct(sentence):
    """Division de mots en texte + suppression de certains caractères"""
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = sentence_clean.split()
    return word_tokens

stop_w = custom_stopwords + ['[', ']', ',', '.', ':', '?', '(', ')','<','>','~']

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

# Prétraitement
df['Cleaned_Body'] = df['Body'].apply(transform_bow_fct)

############################################# Encoding TF - IDF  #############################################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_array = tfidf_vectorizer.fit_transform(df['Cleaned_Body']).toarray()

# Liste des termes correspondant aux colonnes de la matrice
feature_names = tfidf_vectorizer.get_feature_names_out()
# Scores de chaque mot
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

# Concaténez les DataFrames sans utiliser de sample
df.reset_index(drop=True, inplace=True)
df2 = pd.concat([df, tfidf_df], axis=1)


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

# Obtention des trois sujets les plus probables pour chaque ligne de tfidf_df
top_n = 3
top_topics_gensim = lda_model_gensim.get_document_topics(corpus_tfidf_gensim)
top_topics_sklearn = lda_model_sklearn.transform(X).argsort(axis=1)[:, -top_n:]

# Création d'une colonne 'lda_predict' avec la liste des sujets les plus probables
df['lda_predict_gensim'] = [sorted(topics, key=lambda x: x[1], reverse=True)[:top_n] for topics in top_topics_gensim]
df['lda_predict_gensim'] = df['lda_predict_gensim'].apply(lambda x: [topic[0] for topic in x])

df['lda_predict_sklearn'] = [topics for topics in top_topics_sklearn]


############################################# supervised #############################################

import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score


rd.seed(0)

def modelisation(enc,name):
    # Convertir les étiquettes ('Tags') en un format binaire
    mlb = MultiLabelBinarizer()
    #binary_labels = mlb.fit_transform(df['Tags']).tolist()
    # Diviser les données pour l'entraînement et le test
    X_train, X_test, y_train, y_test = train_test_split(enc, df['Tags'], test_size=0.2, random_state=42)
    y_train = [y[rd.randint(0, 2)] for y in y_train]
    y_test = [y[rd.randint(0, 2)] for y in y_test]
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    # Initialisation et entrainement du modèle


    # Prédictions
    predicted_tags = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_tags)

    return("La précision du modèle supervisé avec ",name," est de ",round(accuracy,2)*100," % ")

modelisation(tfidf_df,'tf_idf')