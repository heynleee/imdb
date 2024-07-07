from flask import Flask, request, jsonify
from models.sentiment_analysis import analyze_sentiment
from models.topic_modeling import topic_modeling
from real_time.real_time_analysis import real_time_sentiment_analysis
from real_time.chatbot import chatbot_response
from business.feedback_analysis import analyze_feedback
from business.recommendation_system import train_recommendation_system, get_recommendation

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.get_json()
    review = data.get('review', '')
    sentiment_result = analyze_sentiment(review)
    return jsonify(sentiment_result)

@app.route('/topic_modeling', methods=['POST'])
def analyze_topic():
    data = request.get_json()
    reviews = data.get('reviews', [])
    topic_result = topic_modeling(reviews)
    return jsonify(topic_result)

@app.route('/real_time', methods=['POST'])
def real_time_analysis():
    data = request.get_json()
    review = data.get('review', '')
    sentiment_result = real_time_sentiment_analysis(review)
    return jsonify(sentiment_result)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('input', '')
    response = chatbot_response(user_input)
    return jsonify({"response": response})

@app.route('/feedback', methods=['POST'])
def feedback_analysis():
    data = request.get_json()
    reviews = data.get('reviews', [])
    feedback_result = analyze_feedback(reviews)
    return jsonify(feedback_result)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id', '')
    recommendation_result = get_recommendation(user_id)
    return jsonify(recommendation_result)

if __name__ == '__main__':
    train_recommendation_system()  # 처음 실행 시 추천 시스템 훈련
    app.run(host='0.0.0.0', port=5000)
