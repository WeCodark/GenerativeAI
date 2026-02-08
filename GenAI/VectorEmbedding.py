from gensim.models import Word2Vec

sentences = [
    ['king', 'is', 'a', "royal",'man'],
    ['queen', 'is', 'a', "royal",'woman'],
    ['man', 'is', 'strong'],
    ['woman', 'is', 'wise'],
    ['king','rules','kingdom'],
    ['queen','rules','kingdom'],
    ['king', 'is', 'rich'],
    ['queen', 'is', 'rich']]

model = Word2Vec(
    sentences,
    vector_size=50, # Embedding Dimension
    window=3,
    min_count=1,
    workers=1
)

print("Word Vector for King: ",model.wv['king'])
print("="*50)

# Let's check some similar Words
print(model.wv.most_similar("king",topn=3))


# King - man + Woman
result = model.wv.most_similar(
    positive = ['king','woman'],
    negative = ['man'],
    topn = 3
)

print(result)

king = model.wv['king']
man = model.wv['man']
woman = model.wv['woman']

new_vector = king - man + woman
print(model.wv.similar_by_vector(new_vector))

# To Measure SImiliraies
print(model.wv.similarity('king','queen'))
print(model.wv.similarity('man','woman'))



