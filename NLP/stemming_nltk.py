from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# let's create some sample words
words = ['running', 'runner','ran', 'easily','fairly','fairness', 'studies', 'studying','studied']

# Porter Stemmer
# Simple and most common stemming technique
porter = PorterStemmer()

print(f'Porter Stemmer: {[porter.stem(word) for word in words]}')

# Snowball stemmer
# It is improved version of porter, and it support multiple language
snowball = SnowballStemmer('english')
print(f'Snowball stemmer: {[snowball.stem(word) for word in words]}')

# Lancaster stemmer
# It is very aggressive stemming, it can make the words very short
lancaster = LancasterStemmer()
print(f'Lancaster stemmer: {[lancaster.stem(word) for word in words]}')