import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url,sep = '\t', header = None, names = ['label', 'message'])


# Step 1: We are going to convert Labels: Spam = 1, ham = 0
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})


x_train,x_test, y_train, y_test = train_test_split(df['message'],df['label_num'],test_size=0.2,random_state=42)

# Step 2: Feature Engineering 
tfidf = TfidfVectorizer(stop_words='english')

# Now we will fit transform the training data --> It learn the vocab from the training set and converts it into number
x_train_vec = tfidf.fit_transform(x_train)
# print(x_train.head())
# print(x_train_vec[:5]) # arrays ---> indexing

# let's convert first 5 rows of the matrix to a dense format
tfidf_df = pd.DataFrame(
    x_train_vec[:5].toarray(),
    columns= tfidf.get_feature_names_out()
)

# we are displaying only the columns that have non-zero values for these 5 rows
# print(tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)])


# We are transforming Test Data.
# Very critical. as we need to use only .transform(), Not .fit_transform()
# We must use the vocab learned from x_train
x_test_vec = tfidf.transform(x_test)

print(f'Vocabulay size: {len(tfidf.get_feature_names_out())} unique words')

# Our preprocessing step is done

# Step 3: Build a model

model = MultinomialNB()

model.fit(x_train_vec,y_train)

# Prediction
y_pred = model.predict(x_test_vec)

# print(y_pred)

# Step 4: Evaluation of Model
print(accuracy_score(y_test, y_pred)) # 97 accuracy


# Let's Feed Sample Data
custom_message = [
    "URGENT! you won a lottery of $2000", # spam
    "Hey man are we still going for walk tomorrow morning?" # ham
]

custom_vec = tfidf.transform(custom_message)

prediction = model.predict(custom_vec)

for msg, pred in zip(custom_message, prediction):
    label = "SPAM" if  pred ==1 else "HAM"
    print(f'{label} - {msg}')