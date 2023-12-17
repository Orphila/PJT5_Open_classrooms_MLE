import streamlit as st
import requests
    
def get_prediction(text):
    api_url = "http://localhost:8081/predict"
    response = requests.get(api_url, params={'text': text})

    if response.status_code == 200:
        result = response.json()

        pred_supervised = result.get('results_supervised', 'Valeur_inconnue')
        pred_unsupervised = result.get('results_unsupervised_0', 'Valeur_inconnue')

        return pred_supervised, pred_unsupervised
    else:
        return None, None

# Interface utilisateur Streamlit
def main():
    st.title("StackOverflow Tags Prediction")

    # Zone de texte pour saisir le texte du post
    user_input = st.text_area("Saisissez le texte du post:", "")

    # Bouton pour déclencher la prédiction
    if st.button("Prédire les tags"):
        
        if user_input:
            # Obtenir les prédictions à partir de l'API
            pred_supervised, pred_unsupervised = get_prediction(user_input)

            if pred_supervised and pred_unsupervised:
                st.success("Tags prédits (Supervisé) : {}".format(pred_supervised))
                st.success("Tags prédits (Non supervisé) : {}".format(pred_unsupervised))
            else:
                st.error("Erreur lors de la prédiction. Veuillez réessayer.")


# Exécuter l'application Streamlit
if __name__ == "__main__":
    main()
