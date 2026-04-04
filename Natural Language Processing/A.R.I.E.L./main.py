import random
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.generation import generate_text
from modules.retrieval import retrieve_text

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    query = data['query']
    corpus_path = data['corpus_path']

    relevant_text = retrieve_text(query, corpus_path)
    
    generated_text = generate_text(relevant_text)

    response = {
        'generated_text': generated_text
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
