import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest

def text_summarizer(text, num_sentences=3):
 
    sentences = sent_tokenize(text)
   
    words = word_tokenize(text)
    
  
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    
 
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    
    sent_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 30:
                    if sentence in sent_scores:
                        sent_scores[sentence] += word_freq[word]
                        print("the 1 \n",sent_scores)
                    else:
                        sent_scores[sentence] = word_freq[word]
                        print("the 2\n",sent_scores)
    
    
    summary_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


text = """
Text summarization is the process of distilling the most important information from a source (or sources) to produce an abridged version for a particular user (or users) and task (or tasks). There are two main approaches to automatic text summarization: extractive and abstractive. Extractive summarization involves selecting and concatenating phrases or sentences directly from the source document, while abstractive summarization involves interpreting and paraphrasing sections of the source document.
"""

summary = text_summarizer(text)
print("Summary:")
print(summary)
