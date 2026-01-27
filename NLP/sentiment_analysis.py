from transformers import pipeline

# pip install transformers
# pip install torch

# # We expect answer in the terms of POSITIVE or NEGATIVE
def sentiment_analysis_demo(text):
    classifier = pipeline('sentiment-analysis') # This is the name of task
    result = classifier(text)
    return result

sa = sentiment_analysis_demo("Money can’t buy happiness.")
print(sa)


def zero_shot_classifier_demo():
  classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
  
  text = "I think i will be making pizza tonight"
  labels = ['Technology', 'war','Germany','Cheese']
  result = classifier(text,labels)
  return result

zsc = zero_shot_classifier_demo()
print(zsc)
