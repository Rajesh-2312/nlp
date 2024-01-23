#Demonstrate object standardization such as replace social media slangs from a text.
import re

def standardize_text(text, slang_mapping):
    # Create a regular expression pattern to find slang words
    pattern = re.compile(r'\b(?:%s)\b' % '|'.join(re.escape(key) for key in slang_mapping.keys()), flags=re.IGNORECASE)
    
    # Replace slang words with standard replacements
    standardized_text = pattern.sub(lambda x: slang_mapping[x.group().lower()], text)
    
    return standardized_text

# Example text with social media slangs
text_with_slangs = "OMG! That selfie is so lit, I can't even. BTW, did you see my bae? #Goals"

# Slang mapping for standardization
slang_mapping = {
    'omg': 'oh my goodness',
    'lit': 'excellent',
    'bae': 'significant other',
    'btw': 'by the way'
}

# Standardize the text
standardized_text = standardize_text(text_with_slangs, slang_mapping)

# Print the results
print("Original Text:", text_with_slangs)
print("Standardized Text:", standardized_text)
