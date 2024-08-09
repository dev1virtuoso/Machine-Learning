import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

corpus_path = "/data/corpus.txt"
model_dir = "/models"
dataset_path = "/data/squad_dataset.json"

def train_model(corpus_path):
    # Read the corpus file
    with open(corpus_path, 'r') as file:
        corpus = file.readlines()

    # Create a TF-IDF vectorizer and transform the corpus into a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    return tfidf_vectorizer, tfidf_matrix

def save_model(model_dir, tfidf_vectorizer, tfidf_matrix):
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the TF-IDF vectorizer and matrix as pickle files
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    
    with open(matrix_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)

# Train the model
tfidf_vectorizer, tfidf_matrix = train_model(corpus_path)

# Save the model
save_model(model_dir, tfidf_vectorizer, tfidf_matrix)
