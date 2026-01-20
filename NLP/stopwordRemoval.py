
import nltk # Natual Language toolkit

# below comman is required to run once
nltk.download('stopwords') # It is a corpus(dictionary) where stopwords are saved (is, the, and)

# We have downloaded the stopword dataset
from nltk.corpus import stopwords

text = """This is a great example to demonstrate basic NLP tasks using NLTK Library and also stop word removal."""

# Load English stopwards as a set for faster lookup
stop_words = set(stopwords.words('english'))
# print(f'StopWords: {stop_words}')
# print(len(stop_words))

# We need to convert the Text into lower case

words = text.lower().split()
print(words)

# To remove stopwords -> We are removing those values which are stopword
filtered = [word for word in words if word not in stop_words]
print(f'Filtered Data: {filtered}')