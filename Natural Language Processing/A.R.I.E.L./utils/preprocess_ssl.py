# Copyright (c) 2024 Carson. All rights reserved.

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('stopwords')  # Download the stopwords corpus

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

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
