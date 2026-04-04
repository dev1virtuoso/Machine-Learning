import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

corpus_path = "/data/corpus.txt"
model_dir = "/models"
dataset_path = "/data/squad_dataset.json"

def train_model(corpus_path):
    with open(corpus_path, 'r') as file:
        corpus = file.readlines()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    return tfidf_vectorizer, tfidf_matrix

def save_model(model_dir, tfidf_vectorizer, tfidf_matrix):
    os.makedirs(model_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    
    with open(matrix_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)

tfidf_vectorizer, tfidf_matrix = train_model(corpus_path)

save_model(model_dir, tfidf_vectorizer, tfidf_matrix)
