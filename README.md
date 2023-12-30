# PJT5_Open_classrooms_MLE

Projet n°5 du parcours ingénieur machine learning OpenClassrooms

Le projet vise à améliorer la gestion des tags sur Stack Overflow en proposant automatiquement des tags pertinents lors de la création de nouvelles questions. On va explorer deux approches : non supervisée et supervisée, pour la génération de suggestions de tags et utiliser des méthodes d'extraction de features spécifiques des données textuelles et de différentes techniques de word/sentence embedding.

Étapes Principales :

Extraction de données : Utilisation de l'outil "StackExchange Data Explorer" pour extraire 50 000 questions, en appliquant des filtres pour garantir la qualité des données.
Test de l'API : Évaluation de la pertinence de l'API de Stack Overflow en réalisant une requête sur 50 questions avec le tag "python" et un score > 50.
Modélisation : Mise en œuvre d'approches non supervisées (bag-of-words) et supervisées (Word2Vec, BERT, USE) pour générer des suggestions de tags.
Évaluation : Développement d'une méthode d'évaluation basée sur le taux de couverture des tags, avec une séparation du jeu de données.
MLOps : Utilisation d'outils MLOps comme MLFlow pour le suivi des expérimentations, l'automatisation du pipeline, et l'analyse du "model drift".
Déploiement : Déploiement d'un modèle de proposition de tags sous forme d'API sur le Cloud.

NB : Il y a 3 versions de app.py (l'API flask). LA première (app.py) permet de faire les test en local en utilisant les logs mlflow, 
la seconde (app2.py, la version déployée) récupère seulement les fichiers dont on a besoin 
la troisième (app3.py) ne se sert pas des fichiers et re-entraine les modèles, ce quie st fonctionnel mais beaucoup trop long

