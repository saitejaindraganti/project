import streamlit as st
import pickle
import re, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from nltk.data import find

# Download stopwords on first run (robust for Streamlit Cloud)
try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Load artifacts (expects model.pkl & vectorizer.pkl in same folder)
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_artifacts()

st.set_page_config(page_title="Movie Sentiment", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Type a movie review and press **Predict Sentiment**.")

review = st.text_area("Movie review", height=150)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.success("✅ Positive review")
        else:
            st.error("❌ Negative review")
