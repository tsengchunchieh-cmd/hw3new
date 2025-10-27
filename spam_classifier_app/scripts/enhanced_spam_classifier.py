import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# NLTK data is assumed to be downloaded now.
# No need for try-except blocks here anymore.

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

def get_tokens(text):
    """
    Tokenizes text, similar to the original defs.py, but uses the refined preprocess_text.
    This function is kept for compatibility with the original notebook's structure if needed,
    but preprocess_text is the main function for cleaning.
    """
    return preprocess_text(text).split()

def get_lemmas(text):
    """
    Lemmatizes and removes stop words from tokens, similar to the original defs.py,
    but uses the refined preprocess_text.
    """
    return preprocess_text(text).split()

# Main script logic will go here later
if __name__ == "__main__":
    print("Enhanced Spam Classifier script initialized.")

    # 1. Load Data
    data_path = "C:\\Users\\falle\\Desktop\\hw3new\\spam_classifier_app\\datasets\\sms_spam_no_header.csv"
    sms = pd.read_csv(data_path, sep=',', names=["type", "text"])

    # 2. Data Distribution Visualization
    plt.figure(figsize=(6, 4))
    sns.countplot(x='type', data=sms)
    plt.title('Distribution of Spam vs. Ham Messages')
    plt.show()

    # 3. Apply Preprocessing
    sms['processed_text'] = sms['text'].apply(preprocess_text)

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(sms['processed_text'], sms['type'], test_size=0.3, random_state=42)

    # 5. Feature Extraction (CountVectorizer and TfidfTransformer)
    # Using ngram_range to include bigrams for richer features
    bow_transformer = CountVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2)).fit(X_train)
    X_train_bow = bow_transformer.transform(X_train)
    X_test_bow = bow_transformer.transform(X_test)

    tfidf_transformer = TfidfTransformer().fit(X_train_bow)
    X_train_tfidf = tfidf_transformer.transform(X_train_bow)
    X_test_tfidf = tfidf_transformer.transform(X_test_bow)

    # 6. Model Training (MultinomialNB)
    spam_detector = MultinomialNB().fit(X_train_tfidf, y_train)

    # 7. Prediction and Evaluation
    y_pred = spam_detector.predict(X_test_tfidf)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Word Clouds
    ham_words = ' '.join(sms[sms['type'] == 'ham']['processed_text'])
    spam_words = ' '.join(sms[sms['type'] == 'spam']['processed_text'])

    wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
    wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_ham, interpolation='bilinear')
    plt.title('Word Cloud for Ham Messages')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_spam, interpolation='bilinear')
    plt.title('Word Cloud for Spam Messages')
    plt.axis('off')
    plt.show()

    # 9. Most Frequent Words (Bar Plots)
    def plot_top_n_words(text_data, title, n=20):
        all_words = ' '.join(text_data).split()
        freq_dist = nltk.FreqDist(all_words)
        top_n_words = pd.DataFrame(freq_dist.most_common(n), columns=['Word', 'Frequency'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=top_n_words)
        plt.title(title)
        plt.show()

    plot_top_n_words(sms[sms['type'] == 'ham']['processed_text'], 'Top 20 Most Frequent Words in Ham Messages')
    plot_top_n_words(sms[sms['type'] == 'spam']['processed_text'], 'Top 20 Most Frequent Words in Spam Messages')

    # 10. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 11. ROC Curve
    # For ROC curve, we need probability estimates for the positive class (spam)
    y_prob = spam_detector.predict_proba(X_test_tfidf)[:, 1] # Probability of the 'spam' class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='spam')
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Save the trained model and vectorizers
    model_dir = "C:\\Users\\falle\\Desktop\\hw3new\\spam_classifier_app\\models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bow_transformer, os.path.join(model_dir, 'bow_transformer.joblib'))
    joblib.dump(tfidf_transformer, os.path.join(model_dir, 'tfidf_transformer.joblib'))
    joblib.dump(spam_detector, os.path.join(model_dir, 'spam_detector.joblib'))
    print(f"\nModel and vectorizers saved to {model_dir}")
