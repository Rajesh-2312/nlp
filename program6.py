#Demonstrate Term Frequency Inverse Document Frequency (TF IDF) using python
#scikit-learn used module
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Natural language processing is a subfield of artificial intelligence.",
    "TF-IDF is a technique used in natural language processing for text analysis.",
    "Python is a programming language widely used in data science and machine learning.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency."
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Display TF-IDF matrix as a pandas DataFrame (for better visualization)
import pandas as pd
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print("TF-IDF Matrix:")
print(df_tfidf)
