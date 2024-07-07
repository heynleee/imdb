import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 데이터 로드 및 전처리
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 유저-아이템 매트릭스 생성
def create_user_item_matrix(data):
    user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return user_item_matrix

# 협업 필터링 모델 학습 (K-최근접 이웃 알고리즘)
def train_model(user_item_matrix):
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix_sparse)
    return model

# 영화 추천
def recommend_movies(model, user_item_matrix, user_id, num_recommendations=5):
    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = model.kneighbors(user_item_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=num_recommendations+1)
    recommendations = [user_item_matrix.index[i] for i in indices.flatten()][1:]
    return recommendations

# 전체 파이프라인 실행 함수
def main(filepath, user_id):
    data = load_data(filepath)
    user_item_matrix = create_user_item_matrix(data)
    model = train_model(user_item_matrix)
    recommendations = recommend_movies(model, user_item_matrix, user_id)
    return recommendations

if __name__ == "__main__":
    filepath = 'data/imdb_review.csv'
    user_id = 123  # 예시 사용자 ID
    recommendations = main(filepath, user_id)
    print(f"Recommendations for user {user_id}: {recommendations}")
