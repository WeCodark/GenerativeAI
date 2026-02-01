# Natural Language Processing (NLP) Fundamentals

This module covers the core techniques used to prepare text data for machine learning models, along with advanced processing tasks and practical projects.

## Files & Concepts

### 1. Preprocessing
*   **`tokenization_nltk.py`**: Breaking text into words or sentences using NLTK.
*   **`stemming_nltk.py`**: Reducing words to their root form (e.g., "running" -> "run").
*   **`lemmatization.py`**: Reducing words to their dictionary form (more accurate than stemming).
*   **`stopwordRemoval.py`**: Removing common words (like "the", "is", "in") that add little meaning.

### 2. Feature Extraction
*   **`bagOfWords.py`**: Implementing the **Bag of Words** model using `CountVectorizer`. Converts text into a frequency matrix.
*   **`tf-idf.py`**: Implementing **TF-IDF (Term Frequency-Inverse Document Frequency)**. Highlights important words while downweighting common ones.

### 3. Advanced NLP Tasks
*   **`ner_pos_dependency.py`**: **Named Entity Recognition (NER)** (identifying people, places, orgs), **Part-of-Speech (POS) Tagging**, and **Dependency Parsing**.
*   **`sentiment_analysis.py`**: Analyzing text to determine sentiment (Positive, Negative, Neutral).
*   **`topic_modeling.py`**: extracting abstract topics from a collection of text documents using **LDA (Latent Dirichlet Allocation)**.

### 4. Projects
Practical applications of NLP concepts.
*   **`Projects/customer_feedback_analyzer.py`**: A comprehensive system that analyzes customer reviews. It performs Sentiment Analysis, Entity Extraction, and Topic Modeling to generate a feedback report.
*   **`Projects/text_classification_Manual.py`**: A manual implementation of text classification algorithms to understand the underlying mechanics.

## How to Run
Navigate to the root directory and run any script using Python:

```bash
# Example: Running Bag of Words
python NLP/bagOfWords.py

# Example: Running the Customer Feedback Analyzer Project
python NLP/Projects/customer_feedback_analyzer.py
```

## Notes
PDF notes explaining these concepts in detail are available in the `Notes/` directory.
