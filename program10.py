'''Convert text to vectors (using term frequency) and apply cosine similarity to provide closeness
among two text'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample text data
text1 = "This is a sample text for vectorization."
text2 = "Text vectorization is an important task in natural language processing."

# Combine the text into a list
corpus = [text1, text2]

# Convert text to vectors using term frequency
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Calculate cosine similarity
cosine_sim = cosine_similarity(X, X)

# Print the results
print("Text 1:", text1)
print("Text 2:", text2)
print("Cosine Similarity Matrix:")
print(cosine_sim)
