# Copyright (c) 2024 Carson. All rights reserved.

import random
from sklearn.feature_extraction.text import TfidfVectorizer

# Retrieval module function
def retrieve_text(query, corpus_path):
    # Load the corpus text from the specified path
    with open(corpus_path, 'r') as file:
        corpus = file.readlines()

    # Train or load a pre-trained TF-IDF model
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Use TF-IDF for text matching and retrieve relevant text snippets
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = tfidf_matrix.dot(query_vector.T)
    ranked_indices = similarity_scores.toarray().flatten().argsort()[::-1]
    relevant_text = [corpus[i] for i in ranked_indices]

    # Return the retrieved relevant text snippets for the generation module to use
    return relevant_text

# Generation module function
def generate_text(relevant_text):
    # Randomly select one from the relevant text snippets as the base for generation
    base_text = random.choice(relevant_text)

    # Use the base text directly as the generated text snippet
    generated_text = base_text

    return generated_text

# Example usage
corpus_path = "/data/corpus.txt"
query = "Python programming language"

relevant_text = retrieve_text(query, corpus_path)
generated_text = generate_text(relevant_text)
print("Generated text snippet:")
print(generated_text)
