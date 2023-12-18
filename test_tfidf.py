from imports import tfidf_df
import pandas as pd

def test_models():
    """Vérification de la présence d'outputs dans les modèles"""
    assert isinstance(tfidf_df, pd.DataFrame)
    assert tfidf_df.shape[1] == 1000
