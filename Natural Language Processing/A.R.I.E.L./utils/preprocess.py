# Copyright © 2024 Carson. All rights reserved.

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

# Initialize the set of stopwords and the stemmer
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function for text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
    
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# Test
text = "This is a sample sentence. We will preprocess this text."
preprocessed_text = preprocess_text(text)
print(preprocessed_text)
