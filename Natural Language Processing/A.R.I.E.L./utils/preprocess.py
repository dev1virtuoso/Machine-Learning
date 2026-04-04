import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    
    tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
    
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

text = "This is a sample sentence. We will preprocess this text."
preprocessed_text = preprocess_text(text)
print(preprocessed_text)
