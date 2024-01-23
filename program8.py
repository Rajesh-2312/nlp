#Implement Text classification using naïve bayes classifier and text blob library.

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

# Sample training data for sentiment analysis
train_data = [
    ('I love this product!', 'positive'),
    ('This is a great service.', 'positive'),
    ('The movie was terrible.', 'negative'),
    ('I dislike this feature.', 'negative'),
    ('The book is okay.', 'neutral'),
    ('This restaurant is fantastic!', 'positive'),
    ('I had a bad experience.', 'negative'),
    ('The weather is nice today.', 'neutral')
]

# Train the Naïve Bayes classifier
classifier = NaiveBayesClassifier(train_data)

# Test the classifier with new data
test_sentence = "I enjoy using this app."
classified_label = classifier.classify(test_sentence)

# Print the results
print("Test Sentence:", test_sentence)
print("Predicted Label:", classified_label)
