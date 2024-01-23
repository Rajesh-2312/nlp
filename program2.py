#Perform lemmatization and stemming using python library nltk.
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

def perform_lemmatization_and_stemming(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Perform stemming using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # Perform lemmatization using WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return stemmed_words, lemmatized_words

# Example text
input_text = "The quick brown foxes are jumping over the lazy dogs"

# Perform lemmatization and stemming
stemmed_words, lemmatized_words = perform_lemmatization_and_stemming(input_text)

# Print the results
print("Original Text:", input_text)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)
