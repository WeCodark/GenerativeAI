# pip install sklearn
from sklearn.feature_extraction.text import CountVectorizer

# Sample email data(corpus) - spam vs ham

corpus = [
    "Congratulations! you have won a free lottery ticket. Click here to claim your prize",
    "Dear user, your account has been compromised. Please reset you password immediately",
    "Limited Time Offer! Buy one get one free on all products. Don't miss out!",
    "Hello friend, just wanted to check in and see how you are doing",
    "Reminder: your appointment is scheduled for tomorrow at 10 AM"
]

# Target labels (For ML Classification)
labels = ["spam","ham","spam",'ham','ham']

# To create bag of words model
vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out()) # vocab

print("="*80)


print(matrix.toarray())
print("="*80)


# Let's show word count for first email
print(f'First Email: \n{corpus[0]}\n')
print("Word Frequency:")
print(dict(zip(vectorizer.get_feature_names_out(), matrix.toarray()[0])))