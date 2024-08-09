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
    # Vectorize the query text
    query_vector = tfidf_vectorizer.transform([query])

    # Compute the similarity between the query text and the texts in the corpus
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Find the index of the most similar text
    most_similar_index = similarities.argmax()

    # Get the most similar text
    most_similar_text = tfidf_matrix[most_similar_index]

    # Convert the most similar text to a string
    generated_text = most_similar_text.toarray().squeeze().tolist()

    return generated_text

# Load the model
tfidf_vectorizer, tfidf_matrix = load_model(model_dir)

# Generate text
generated_text = generate_text(query, tfidf_vectorizer, tfidf_matrix)

# Print the generated text
print("Generated Text:", generated_text)
