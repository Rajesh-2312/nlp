#Perform part of speech tagging on any textual data.
#requirement nltk

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def perform_pos_tagging(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(words)
    
    return pos_tags

# Example text
input_text = "The quick brown foxes are jumping over the lazy dogs"

# Perform part-of-speech tagging
pos_tags = perform_pos_tagging(input_text)

# Print the results
print("Original Text:", input_text)
print("Part-of-Speech Tags:", pos_tags)
