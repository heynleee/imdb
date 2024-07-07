# topic_modeling.py

from gensim import corpora, models
import pandas as pd

def perform_topic_modeling(texts):
    # 주제 모델링 코드
    texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)
    # 주제 출력
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx}, Words: {topic}")
