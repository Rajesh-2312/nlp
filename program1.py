'''Demonstrate Noise Removal for any textual data and remove regular expression pattern such as
hash tag from textual data.'''
import re

def remove_noise(text):
    # Remove hash tags
    text = re.sub(r'#\w+', '', text)
    
    # Remove non-alphabetic characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example text with hash tags and noise
text_with_noise = "This is a #sample text with @some noise and extra   spaces! #Demo"

# Remove noise
cleaned_text = remove_noise(text_with_noise)

# Print the cleaned text
print("Original Text:", text_with_noise)
print("Cleaned Text:", cleaned_text)
