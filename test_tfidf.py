from imports import tfidf_df
import pandas as pd

def test_tfidf():
    """ Vérification que l'encodage a bien 1000 colonnes """
    assert isinstance(tfidf_df, pd.DataFrame)
    assert tfidf_df.shape[1] == 1000
