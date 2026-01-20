from nltk.tokenize import word_tokenize, sent_tokenize

import nltk

# nltk.download('punkt_tab') # Only once
# NLTK use a pre trained tokenization model
# punkt_tab contains rules and data needed to split the text into senteces and words

# input text
text = "Hello! How are you doing? I am Learning NLP. It's amazing."

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentece Tokens: ")
print(sentences)
# Here in our sample text data the sentece are identified by punctuation marks

# Word Tokenization
words = word_tokenize(text)
print("\nWord Tokens: ")
print(words)

# Character tokeniation
text = 'Hello Aryan!'

char = list(text.lower())
print(char)