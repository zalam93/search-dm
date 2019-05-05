#!flask/bin/python
import numpy as np
import pandas as pd
import re
import operator
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
        query_tfidf = query_tfidf.tocoo()
        sim_score = cosine_similarity(query_tfidf, train_tfidf)
        sorted_indexes = np.argsort(sim_score).tolist()
        company_indexes = (dataset['Company'].iloc[sorted_indexes[0][-10:]].drop_duplicates()).tolist()
        num = len(company_indexes) * -1
        sorted_scores = sorted_indexes[0][num:]
        return company_indexes, sorted_scores, query_tfidf.data
    company, sorted_scores, query_tfidf = get_search_results(query)
    # scores = ','.join(str(i) for i in sorted_scores)
    '''scores = scores.split(',')

    json = [{
        'scores': scores,
        'company': company


    }]'''
    print(query_tfidf)

    return company, sorted_scores, query_tfidf


def index2(query):

    val = query.split(',')
    data = pd.read_csv('employee_reviews.csv')
    data = data[['Company', 'Rating', 'Workload', 'Culture', 'Growth', 'Benefits', 'Management']]
    n_rows = data['Company'].count()

    def count(name):
        n_count = data['Company'][data['Company'] == name].count()
        return n_count

    def P_count(name):
        P_company = count(name) / n_rows
        return P_company

    data_means = data.groupby('Company').mean()
    data_var = data.groupby('Company').var()

    def calc_mean(col, name):
        mean = data_means[col][data_var.index == name].values[0]
        return mean

    def calc_var(col, name):
        mean = data_var[col][data_var.index == name].values[0]
        return mean

    def Px_given_Py(x, my, vy):
        p = 1 / (np.sqrt(2 * np.pi * vy)) * np.exp((-(x - my)**2) / (2 * vy))
        return p

    # calculating results for each company

    def prediction(c0, c1, c2, c3, c4, c5, name):
        result = P_count(name) * \
            Px_given_Py(c0, calc_mean('Workload', name), calc_var('Rating', name)) * \
            Px_given_Py(c1, calc_mean('Workload', name), calc_var('Workload', name)) * \
            Px_given_Py(c2, calc_mean('Culture', name), calc_var('Culture', name)) * \
            Px_given_Py(c3, calc_mean('Growth', name), calc_var('Growth', name)) * \
            Px_given_Py(c4, calc_mean('Benefits', name), calc_var('Benefits', name)) * \
            Px_given_Py(c5, calc_mean('Management', name), calc_var('Management', name))
        return result

    google = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Google')
    amazon = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Amazon')
    apple = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Apple')
    facebook = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Facebook')
    netflix = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Netflix')
    microsoft = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Microsoft')
    ibm = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'IBM')
    deloitte = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'Deloitte')
    airbnb = prediction(int(val[0]), int(val[1]), int(val[2]), int(val[3]), int(val[4]), int(val[5]), 'AirBnb')
    companies = {'Google': google, 'Amazon': amazon, 'Apple': apple, 'Facebook': facebook,
                 'Netflix': netflix, 'Microsoft': microsoft, 'IBM': ibm, 'Deloitte': deloitte, 'AirBnb': airbnb}

    return str(max(companies.items(), key=operator.itemgetter(1))[0]).strip()
