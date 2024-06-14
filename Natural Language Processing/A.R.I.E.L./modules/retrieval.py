# Copyright © 2024 Carson. All rights reserved.

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

# Example usage
corpus_path = "/data/corpus.txt"
query = "Python programming language"

relevant_text = retrieve_text(query, corpus_path)
print("Relevant text snippets retrieved:")
for text in relevant_text:
    print(text)
