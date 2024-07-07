import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

data = load_data('data/imdb_review.csv')
