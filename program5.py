#Implement topic modeling using Latent Dirichlet Allocation (LDA ) in python.
#requirement gensim
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample documents
documents = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Topic modeling is a technique used in natural language processing.",
    "Latent Dirichlet Allocation is a popular algorithm for topic modeling.",
    "Python is a programming language widely used in data science and natural language processing."
]

def preprocess_text(doc):
    # Tokenize and lowercase
    tokens = word_tokenize(doc.lower())
    
    # Remove punctuation and stop words
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return lemmatized_tokens

# Preprocess the documents
preprocessed_docs = [preprocess_text(doc) for doc in documents]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(preprocessed_docs)

# Create a corpus from the documents
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Build the LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print the topics and their top words
print("Topics:")
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
