# Amazon Product Review Sentiment Analysis

A machine learning project that predicts whether an Amazon product review is **Positive** or **Negative** based on customer feedback text.

---

## Project Overview
This project performs sentiment analysis on **500,000+ Amazon product reviews** using **Natural Language Processing (NLP)** and **Machine Learning**.  
Text data is cleaned, vectorized using **TF-IDF**, and classified using **Logistic Regression**, achieving **87% accuracy** on test data.

---

## Features
- Preprocess text data (cleaning, stopword removal, lemmatization)
- Convert text to numerical features using **TF-IDF (5000 features)**
- Train and evaluate a **Logistic Regression classifier**
- Save and reuse trained model & vectorizer for predictions
- Predict sentiment for any custom input text

---

## Tech Stack
| Component | Tools |
|----------|-------|
| Language | Python |
| NLP | NLTK, SpaCy |
| ML | Scikit-learn |
| Data Handling | Pandas, NumPy |

---

## 📂 Project Structure
amazon-sentiment-analysis/
│
├── data/ # Dataset folder (dataset not uploaded due to size)
│ └── README.md # Instructions to download dataset
│
├── model/ # Saved model files after training
│ ├── sentiment_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── notebooks/
│ └── sentiment_analysis.ipynb # Exploratory analysis & experimentation
│
├── src/
│ ├── preprocessing.py # Text preprocessing functions
│ ├── train_model.py # Training script
│ └── predict.py # Model loading & prediction script
│
├── results/
│ └── metrics.txt # Evaluation metrics
│
├── requirements.txt
└── README.md
