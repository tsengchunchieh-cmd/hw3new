# Spam Classifier Project Summary

This document summarizes a project for building and running a spam classifier, based on the provided `PacktPublishing.pdf` and `sms_spam_no_header.csv` files.

## Project Overview

The project aims to build a machine learning model to classify SMS messages as "spam" or "ham" (not spam). It uses a dataset of SMS messages and a series of Python scripts to preprocess the data, train a model, and make predictions. The project also includes tools for visualizing the results and an interactive web application for real-time classification.

## Dataset

The raw data for this project is provided in the `sms_spam_no_header.csv` file. It is a CSV file with two columns:

*   **Column 1: Label**: Indicates whether the message is "ham" or "spam".
*   **Column 2: Message**: The text of the SMS message.

## Project Steps

The following steps outline the process of building and running the spam classifier, as described in the PDF:

### 1. Preprocessing the Data

This step cleans the raw SMS data and prepares it for training.

```bash
python scripts/preprocess_emails.py \
--input datasets/sms_spam_no_header.csv \
--output datasets/processed/sms_spam_clean.csv \
--no-header --label-col-index 0 --text-col-index 1 \
--output-text-col text_clean \
--save-step-columns \
--steps-out-dir datasets/processed/steps
```

### 2. Training the Classifier

This step uses the preprocessed data to train a classification model.

```bash
python scripts/train_spam_classifier.py \
--input datasets/processed/sms_spam_clean.csv \
--label-col col_0 --text-col text_clean \
--class-weight balanced \
--ngram-range 1,2 \
--min-df 2 \
--sublinear-tf \
--C 2.0 \
--eval-threshold 0.50
```

### 3. Making Predictions

This step uses the trained model to predict whether a new message is spam or ham.

```bash
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"
```

### 4. Visualizing the Results

The project includes scripts to visualize various aspects of the data and model performance, such as:

*   Class distribution
*   Token frequency
*   Confusion matrix, ROC, and PR curves
*   Threshold sweep

```bash
python scripts/visualize_spam.py \
--input datasets/processed/sms_spam_clean.csv \
--label-col col_0 \
--class-dist
```

### 5. Interactive Streamlit App

The project can be run as an interactive web application using Streamlit.

```bash
streamlit run app/streamlit_app.py
```

This provides a user-friendly interface for classifying messages in real-time.
