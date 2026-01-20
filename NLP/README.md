# Natural Language Processing (NLP) Fundamentals 🧠

This module covers the core techniques used to prepare text data for machine learning models.

## 📂 Files & Concepts

### 1. Preprocessing
*   **`tokenization_nltk.py`**: Breaking text into words or sentences.
*   **`stemming_nltk.py`**: Reducing words to their root form (e.g., "running" -> "run").
*   **`lemmatization.py`**: Reducing words to their dictionary form (more accurate than stemming).
*   **`stopwordRemoval.py`**: Removing common words (like "the", "is", "in") that add little meaning.

### 2. Feature Extraction
*   **`bagOfWords.py`**: Implementing the **Bag of Words** model using `CountVectorizer`. Converts text into a frequency matrix.
*   **`tf-idf.py`**: Implementing **TF-IDF (Term Frequency-Inverse Document Frequency)**. Highlights important words while downweighting common ones.

## 🚀 How to Run
Navigate to the root directory and run any script using Python:

```bash
# Example: Running Bag of Words
python NLP/bagOfWords.py

# Example: Running TF-IDF
python NLP/tf-idf.py
```

## 📝 Notes
PDF notes explaining these concepts in detail are available in the `Notes/` directory.
