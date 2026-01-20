from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# sample document corpus
documents = [
    "the cat sat on the mat",   # doc 1
    "the dog sat on the log",   # doc 2 
    "cats and dogs are animals" # doc 3
]

tfidifVec = TfidfVectorizer()

matrix = tfidifVec.fit_transform(documents)

# print(matrix.toarray())

df = pd.DataFrame(
    matrix.toarray(),
    columns= tfidifVec.get_feature_names_out(),
    index=['Doc1','Doc2','Doc3']
)

print(df.round(2))