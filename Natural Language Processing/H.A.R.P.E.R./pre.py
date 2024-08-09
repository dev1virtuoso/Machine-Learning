import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stopwords_list = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stopwords_list]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the preprocessed tokens back into a single string
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text

# Specify the file path
file_path = '/path/to/directory'

# Preprocess the text file
preprocessed_text = preprocess_text_file(file_path)

# Print the preprocessed text
print(preprocessed_text)

def write_preprocessed_text(file_path, preprocessed_text):
    with open(file_path, 'w') as file:
        file.write(preprocessed_text)

# Specify the file path to write the preprocessed text
output_file_path = '/path/to/directory'

# Write the preprocessed text to the file
write_preprocessed_text(output_file_path, preprocessed_text)
