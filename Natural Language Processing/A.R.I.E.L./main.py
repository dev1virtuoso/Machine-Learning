import random
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.generation import generate_text
from modules.retrieval import retrieve_text

app = Flask(__name__)

# Define API route
@app.route('/generate', methods=['POST'])
def generate():
    # Get data from request
    data = request.get_json()
    query = data['query']
    corpus_path = data['corpus_path']

    # Retrieve relevant text based on the query from the corpus
    relevant_text = retrieve_text(query, corpus_path)
    
    # Generate new text based on the relevant text
    generated_text = generate_text(relevant_text)

    # Create response JSON
    response = {
        'generated_text': generated_text
    }
    return jsonify(response)

# Start the API server
if __name__ == '__main__':
    app.run()
