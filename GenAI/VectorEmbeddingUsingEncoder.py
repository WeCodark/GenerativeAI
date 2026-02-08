# pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

from sentence_transformers.util import cos_sim

# Here we are loading the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Words to compare
words = ['king','queen']

# We will try to get the embeddings for these words
embeddings = model.encode(words)


# Now we will compute cosine Similarity
similarity = cos_sim(embeddings[0],embeddings[1])

print(f'Similarity: {similarity}')
print(f'king: {embeddings[0]}')
print(f'queen: {embeddings[1]}')