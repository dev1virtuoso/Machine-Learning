import os
import pickle
from modules.generation import generate_text
from modules.retrieval import retrieve_text

corpus_path = "/data/corpus.txt"
model_dir = "/models"

def generate(query, model_dir):
    # Load the TF-IDF vectorizer and matrix from the saved model
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    
    with open(vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    
    with open(matrix_path, 'rb') as file:
        tfidf_matrix = pickle.load(file)
    
    # Retrieve relevant text based on the query from the corpus
    relevant_text = retrieve_text(query, corpus_path)
    
    # Generate new text based on the relevant text
    generated_text = generate_text(relevant_text)

    return generated_text

# Generate text
query = "The query you want to generate"
generated_text = generate(query, model_dir)

print("Generated text:", generated_text)
