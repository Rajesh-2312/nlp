#Demonstrate word embeddings using word2vec
#required module is gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample sentences
sentences = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Word embeddings capture semantic relationships between words.",
    "Python is a popular programming language for machine learning and data science."
]

# Tokenize the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector representation of a word
word_vector = model.wv['language']
print("Vector representation of 'language':", word_vector)

# Find similar words
similar_words = model.wv.most_similar('natural', topn=3)
print("Words similar to 'natural':", similar_words)



'''

1.We tokenize the input sentences using nltk.word_tokenize.
2.We create a Word2Vec model using the Word2Vec class from gensim.
3.The vector_size parameter specifies the dimensionality of the word vectors.
4.The window parameter determines the maximum distance between the current and predicted word within a sentence.
5.The min_count parameter specifies the minimum number of occurrences of a word to be considered.
6.The workers parameter controls the number of CPU cores used for training.
After training the Word2Vec model, you can access the vector representation of a word (word_vector) and find similar words to a given word (similar_words).

Make sure to adjust the parameters based on your specific requirements and the size of your dataset. Additionally, you might want to train the model on a larger corpus for better word embeddings in practice.

'''