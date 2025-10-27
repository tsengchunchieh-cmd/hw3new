import streamlit as st
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# NLTK data is assumed to be downloaded now.
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Converts text to lowercase, removes non-alphabetic characters,
    tokenizes, removes stop words, and lemmatizes.
    """
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text.lower())
    
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Remove stop words and lemmatize
    lemmas = [wordnet_lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return " ".join(lemmas)

# Load the model artifacts
model_dir = "C:\\Users\\falle\\Desktop\\hw3new\\spam_classifier_app\\models"
bow_transformer = joblib.load(os.path.join(model_dir, 'bow_transformer.joblib'))
tfidf_transformer = joblib.load(os.path.join(model_dir, 'tfidf_transformer.joblib'))
spam_detector = joblib.load(os.path.join(model_dir, 'spam_detector.joblib'))

st.title("Spam Classifier App")
st.write("Enter a message below to classify it as spam or ham.")

user_input = st.text_area("Enter your message here:", "")

if st.button("Classify"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_bow = bow_transformer.transform([processed_input])
        input_tfidf = tfidf_transformer.transform(input_bow)
        prediction = spam_detector.predict(input_tfidf)[0]
        
        st.write(f"The message is: **{prediction.upper()}**")
    else:
        st.write("Please enter a message to classify.")