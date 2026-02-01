# Topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

documents = [
    "Python programming is great for data science and machine learning",
    "I love playing basketball and football with my friends",
    "Cooking pasta with tomato sauce is delicious",
    "Machine learning algorithms need lots of data to perform well",
    "Soccer and tennis are popular sports worlwide",
    "Coding is a good habbit",
    'Baking cakes required flour, sugar, and eggs',
    "Deep Learning use neural network to model complex pattern in data",
    "swimming and running keep you healthy and fit",
    "French cuisine includes dishes like croissants and ratatouille"
]

print("="*40)
print('Topic Modeling Demo')
print("="*40)

print(f'We have {len(documents)} documents')

# Step 1: Convert text data into numbers
vectorizer = CountVectorizer(stop_words='english') # stop word here it supports only english
doc_matrix = vectorizer.fit_transform(documents)
print(f'Document-Term Matrix Shape: {doc_matrix.shape}')

# Step 2: find the topics - Top 3 
num_topics = 3
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(doc_matrix)

# Step 3: Visualize topics
words = vectorizer.get_feature_names_out()
print('='*40)
print(words)
print('='*40)

for topic_num, topic in enumerate(lda_model.components_):
    top_words_indices = topic.argsort()[-5:][::-1] # Top 5 words for each topic
    top_words = [words[i] for i in top_words_indices]
    print(f'Topic {topic_num + 1}: {','.join(top_words)}')


# Classification - New Document
new_docs = ["I enjoy coding in python and building Machine Learning Models",
            "Baking bread and pastries is my favourite hobby",
            "I enjoy coding in Java and eat a lots of pasta"]

print('='*40)
print('New Document classification')
print('='*40)
for i in new_docs:
    new_matrix = vectorizer.transform([i]) 
    topic_probs = lda_model.transform(new_matrix)[0]

    print(f'Document: {i}')
    print(f'Topic Probability:')
    for i, prob in enumerate(topic_probs):
        print(f'==== Topic {i+1}: {prob:.4f}')

