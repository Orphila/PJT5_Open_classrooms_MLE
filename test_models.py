from imports import lda_model_sklearn, modelisation, tfidf_df


def test_models():
    """Vérification de la présence d'outputs dans les modèles"""
    assert lda_model_sklearn.n_components == 50
    assert isinstance(modelisation(tfidf_df, 'tf_idf')[-2], float)
