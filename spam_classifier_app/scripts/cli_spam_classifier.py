import joblib
import os
import sys
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

def classify_text(text, bow_transformer, tfidf_transformer, spam_detector):
    processed_text = preprocess_text(text)
    text_bow = bow_transformer.transform([processed_text])
    text_tfidf = tfidf_transformer.transform(text_bow)
    prediction = spam_detector.predict(text_tfidf)[0]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_spam_classifier.py \"Your text here\"")
        sys.exit(1)

    # Load the model artifacts
    model_dir = "C:\\Users\\falle\\Desktop\\hw3new\\spam_classifier_app\\models"
    bow_transformer = joblib.load(os.path.join(model_dir, 'bow_transformer.joblib'))
    tfidf_transformer = joblib.load(os.path.join(model_dir, 'tfidf_transformer.joblib'))
    spam_detector = joblib.load(os.path.join(model_dir, 'spam_detector.joblib'))

    input_text = sys.argv[1]
    prediction = classify_text(input_text, bow_transformer, tfidf_transformer, spam_detector)

    print(f"The text \"{input_text}\" is classified as: {prediction}")
