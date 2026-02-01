# NLP Projects

This directory contains practical projects that apply various Natural Language Processing techniques.

## Project List

### 1. Customer Feedback Analyzer
*   **File**: `customer_feedback_analyzer.py`
*   **Description**: A comprehensive tool designed to analyze customer feedback or reviews.
*   **Key Features**:
    *   **Sentiment Analysis**: Determines if a review is Positive, Negative, or Neutral.
    *   **Entity Extraction**: Identifies key entities like Products, Organizations, and People mentioned in the text.
    *   **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to discover underlying themes (topics) in the feedback.
    *   **Statistical Summary**: Provides a breakdown of positive vs. negative reviews and top keywords.
*   **Libraries Used**: `spacy`, `sklearn`, `transformers`.

### 2. Manual Text Classification
*   **File**: `text_classification_Manual.py`
*   **Description**: A stripped-down, manual implementation of text classification algorithms.
*   **Purpose**: To understand the internal mechanics of classification algorithms (like Naive Bayes) without relying on high-level library abstractions.
*   **Key Concepts**: Probability calculation, feature selection, and model training from scratch.

## Usage

Run the projects from the root directory:

```bash
# Run Customer Feedback Analyzer
python NLP/Projects/customer_feedback_analyzer.py

# Run Manual Text Classification
python NLP/Projects/text_classification_Manual.py
```
