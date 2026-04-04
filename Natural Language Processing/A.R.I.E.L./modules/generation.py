import random
from sklearn.feature_extraction.text import TfidfVectorizer

def retrieve_text(query, corpus_path):
    with open(corpus_path, 'r') as file:
        corpus = file.readlines()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = tfidf_matrix.dot(query_vector.T)
    ranked_indices = similarity_scores.toarray().flatten().argsort()[::-1]
    relevant_text = [corpus[i] for i in ranked_indices]

    return relevant_text

def generate_text(relevant_text):
    base_text = random.choice(relevant_text)

    generated_text = base_text

    return generated_text

corpus_path = "/data/corpus.txt"
query = "Python programming language"

relevant_text = retrieve_text(query, corpus_path)
generated_text = generate_text(relevant_text)
print("Generated text snippet:")
print(generated_text)
