# Import spaCy library for NLP tasks
import spacy
from spacy import displacy

# Load English language model
# (Run: spacy.cli.download('en_core_web_sm') once if not installed)
nlp = spacy.load('en_core_web_sm')


# Example 1: Basic NLP Pipeline

# This text will be processed by spaCy NLP pipeline
text = "Apple CEO Tim Cook announced a new iPhone in California on Monday"
doc = nlp(text)


# POS Tagging (Part of Speech)

# POS tells us what role each word plays in the sentence
for token in doc:
    print(token.text, token.pos_, token.tag_)

# token.text  -> actual word
# token.pos_  -> coarse POS (NOUN, VERB, PROPN)
# token.tag_  -> fine-grained POS (NNP, VBD, etc.)


# POS Tagging with Explanation Table

# This shows POS details in a readable table format
text = "I had a flight from NewYork to San Francisco on 24th January 2026"
doc = nlp(text)

print(f"{'Word':<15} {'Lemma':<15} {'POS':<10} {'Tag':<10} Explanation")
print("=" * 80)

for token in doc:
    explanation = spacy.explain(token.tag_) or "N/A"
    print(f"{token.text:<15} {token.lemma_:<15} {token.pos_:<10} {token.tag_:<10} {explanation}")


# Named Entity Recognition (NER)

# NER finds real-world entities like person, location, organisation
text = "Apple CEO Tim Cook announced a new iPhone in California on Monday"
doc = nlp(text)

# Print entity text and entity type
for ent in doc.ents:
    print(f"{ent.text:<15} {ent.label_}")



# NER using IOB Tagging (B-I-O)

# B = Beginning, I = Inside, O = Outside of entity
for token in doc:
    print(f"{token.text:<15} {token.ent_iob_:<5} {token.ent_type_}")



# Dependency Parsing Example

# Dependency parsing shows grammatical relationship between words
text = "I ate and then slept after heavy dinner."
doc = nlp(text)

print(f"{'Word':<15} {'Relation':<10} {'Head'}")
print("=" * 80)

for token in doc:
    print(f"{token.text:<15} {token.dep_:<10} {token.head.text}")



# Visualising Dependency Parse Tree

# This creates an HTML file to view dependency tree in browser
html = displacy.render(doc, style='dep')

with open('dependencyTree.html', 'w', encoding='utf-8') as f:
    f.write(html)
