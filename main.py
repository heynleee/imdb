# main.py

import os
import pandas as pd
from business.feedback_analysis import preprocess_data, train_sentiment_model, predict_sentiment
from business.recommendation_system import main as recommendation_system_main
from models.topic_modeling import perform_topic_modeling
from real_time.real_time_analysis import analyze_reviews_real_time

# 데이터 파일 경로
DATA_FILE_PATH = 'data/imdb_review.csv'

def main():
    # 1. 데이터 로드 및 전처리
    if os.path.exists(DATA_FILE_PATH):
        data = pd.read_csv(DATA_FILE_PATH)
    else:
        raise FileNotFoundError(f"{DATA_FILE_PATH} not found.")
    
    preprocessed_data = preprocess_data(data)
    
    # 2. 감정 분석 모델 학습 및 예측
    X, y = preprocessed_data['review_text'], preprocessed_data['sentiment']
    model, vectorizer = train_sentiment_model(X, y)
    preprocessed_data['predicted_sentiment'] = predict_sentiment(model, vectorizer, X)
    
    # 3. 주제 모델링 수행
    perform_topic_modeling(preprocessed_data['review_text'])
    
    # 4. 추천 시스템 실행 (예시 사용자 ID로 123 사용)
    user_id = 123
    recommendations = recommendation_system_main(DATA_FILE_PATH, user_id)
    print(f"Recommendations for user {user_id}: {recommendations}")
    
    # 5. 실시간 리뷰 분석 (예시 데이터 사용)
    analyze_reviews_real_time(preprocessed_data['review_text'])

if __name__ == "__main__":
    main()
