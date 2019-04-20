#!flask/bin/python
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def index(query):
    dataset = pd.read_csv('employee_reviews.csv')
    feature = dataset['Pros']
    stop_words = stopwords.words('english')

    def process_text(feature):
        feature = str(feature)
        feature = re.sub('[^a-zA-Z\s]', '', feature)
        feature = [w for w in feature.split() if w not in set(stop_words)]
        return ' '.join(feature)

    feature = dataset['Pros'].apply(process_text)
    english_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()

    def stemming(feature):
        return (english_stemmer.stem(w) for w in analyzer(feature))

    count = CountVectorizer(analyzer=stemming)
    count_matrix = count.fit_transform(feature)
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(count_matrix)

    def get_search_results(query):
        query = process_text(query)
        query_matrix = count.transform([query])
        query_tfidf = tfidf_transformer.transform(query_matrix)
        sim_score = cosine_similarity(query_tfidf, train_tfidf)
        sorted_indexes = np.argsort(sim_score).tolist()
        company_indexes = (dataset ['Company'].iloc[sorted_indexes[0][-10:]].drop_duplicates()).tolist()
        num = len(company_indexes) * -1
        sorted_scores = sorted_indexes[0][num:]
        return company_indexes, sorted_scores

    company, sorted_scores = get_search_results(query)
    scores = ','.join(str(i) for i in sorted_scores)
    scores = scores.split(',')

    json = [{
        'scores': scores,
        'company': company


    }]

    return json
