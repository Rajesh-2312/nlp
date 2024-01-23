#Apply support vector machine for text classification.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample training data for sentiment analysis
corpus = [
    ('I love this product!', 'positive'),
    ('This is a great service.', 'positive'),
    ('The movie was terrible.', 'negative'),
    ('I dislike this feature.', 'negative'),
    ('The book is okay.', 'neutral'),
    ('This restaurant is fantastic!', 'positive'),
    ('I had a bad experience.', 'negative'),
    ('The weather is nice today.', 'neutral')
]

# Separate text and labels
X, y = zip(*corpus)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
