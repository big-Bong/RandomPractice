import string
import collections
import os, sys
#import pandas as pd
 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
 
 
def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    #transtab = string.maketrans(string.punctuation,'')
    text = text.strip(string.punctuation)
    stopwords_set = set(stopwords.words('english'))
    text = ' '.join([val for val in text.split() if val not in stopwords_set])
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(texts[idx])
 
    return clustering
 
 
if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    file = open(os.path.join(curr_dir,"dataset.txt"),'r')
    articles = []
    #print(string.punctuation)
    text = file.readline()
    while(text):
        articles.append(text)
        text = file.readline()
    clusters = cluster_texts(articles, 10)
    pprint(dict(clusters))