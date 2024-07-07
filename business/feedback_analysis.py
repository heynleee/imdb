# feedback_analysis.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def preprocess_data(data):
    # 데이터 전처리 코드
    data['review_text'] = data['review_text'].str.lower()
    # 추가 전처리 코드
    return data

def train_sentiment_model(X, y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_vec, y)
    joblib.dump(model, 'models/sentiment_analysis_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_sentiment(model, vectorizer, X):
    X_vec = vectorizer.transform(X)
    predictions = model.predict(X_vec)
    return predictions
