import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    stopwords_list = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stopwords_list]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text

file_path = '/path/to/directory'

preprocessed_text = preprocess_text_file(file_path)

print(preprocessed_text)

def write_preprocessed_text(file_path, preprocessed_text):
    with open(file_path, 'w') as file:
        file.write(preprocessed_text)

output_file_path = '/path/to/directory'

write_preprocessed_text(output_file_path, preprocessed_text)
