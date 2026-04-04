import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model_dir = "/models"
query = "Hi"

def load_model(model_dir):
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")

    with open(vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    with open(matrix_path, 'rb') as file:
        tfidf_matrix = pickle.load(file)

    return tfidf_vectorizer, tfidf_matrix

def generate_text(query, tfidf_vectorizer, tfidf_matrix):
    query_vector = tfidf_vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)

    most_similar_index = similarities.argmax()

    most_similar_text = tfidf_matrix[most_similar_index]

    generated_text = most_similar_text.toarray().squeeze().tolist()

    return generated_text

tfidf_vectorizer, tfidf_matrix = load_model(model_dir)

generated_text = generate_text(query, tfidf_vectorizer, tfidf_matrix)

print("Generated Text:", generated_text)
