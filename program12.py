'''Detect hate speech in tweets.
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a
tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task
is to classify racist or sexist tweets from other tweets.'''


import pandas as pd
import nltk
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('program11.csv')




nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['tweet'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)



vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)




svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)





y_pred = svm_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


new_tweets = ["I love my new phone!", "This is an example of hate speech."]
new_tweets_processed = [preprocess_text(tweet) for tweet in new_tweets]
new_tweets_tfidf = vectorizer.transform(new_tweets_processed)

predicted_labels = svm_classifier.predict(new_tweets_tfidf)
print("Predicted Labels:", predicted_labels)
