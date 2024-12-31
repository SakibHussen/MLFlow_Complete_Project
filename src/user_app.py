# Import dependencies
import streamlit as st
import mlflow
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
mlflow.set_tracking_uri("http://localhost:5001")
#run_id='d4fa1636f19c45ef9f62c37b605aa01c'
# Helper Functions
def get_production_model():
    try:
        # This will fetch all registered models
        prod = [model for model in mlflow.search_model_versions() if model.name == 'Spamfilter' and model.current_stage == 'Production']
        model = prod[0]
        return model
    except:
        print("No Production Model Found")

model = get_production_model()
# Run_ID='241d005a3872498182883d60d33ffb9d'
# #Loading vector and model
# logged_vect = f'runs:/{Run_ID}/vectorizer'
# logged_model = f'runs:/{Run_ID}/model'


logged_vect = f'runs:/{model.run_id}/vectorizer'
logged_model = f'runs:/{model.run_id}/model'

def preprocess_text(text):
    # For preprocessing the text: Removing stopwords and then Stemming it
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_words)

# Sidebar Information
st.sidebar.title("About Me 🧑‍💻")
try:
    st.sidebar.header(f"Model Name:\n```{model.name}```")
    st.sidebar.header(f"Model Version:\n```{model.version}```")
    st.sidebar.header(f"Current Stage:\n```{model.current_stage}```")
    st.sidebar.subheader(f"Run ID:\n```{model.run_id}```")
    if 'loaded_model' not in st.session_state:
        with st.spinner("Loading Models"):
            st.session_state['loaded_model'] = mlflow.pyfunc.load_model(logged_model)
            st.session_state['loaded_vect'] = mlflow.sklearn.load_model(logged_vect)
    st.sidebar.success("Server is Up & Models are loaded 🔥")
except:
    st.sidebar.warning("Models not found")

# Main Area
st.title("Spam Filter 📬 🛡️")
# Making Predictions
text = st.text_area("Enter your Message / Email in the box below 👇")
if st.button("Classify 🚀"):
    processed_text = preprocess_text(text)
    vectorized_text = st.session_state['loaded_vect'].transform([processed_text])
    prediction = st.session_state['loaded_model'].predict(vectorized_text)
    if prediction[0] == 'spam':
        st.subheader(f"Looks like a Spam ❌")
    else:
        st.subheader(f"Looks Safe ✅")
