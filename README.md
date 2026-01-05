# Aspect-Based Sentiment Analysis for E-commerce Reviews

## Overview
Built an aspect-level sentiment analysis pipeline for e-commerce product reviews, achieving 85% weighted F1 score on a custom-labeled dataset.

## Problem
Traditional sentiment analysis provides overall sentiment, but e-commerce reviews often contain mixed sentiments (e.g., "love the product but hate the shipping"). This project extracts aspect-specific sentiment to provide granular insights.

## Approach

### Dataset Creation
- Started with 43,000 unlabeled women's clothing reviews
- Used spaCy for aspect extraction (noun chunks)
- Applied DistilBERT for baseline sentiment classification
- Refined labels using GPT-3.5 Turbo (temperature=0) for aspect-specific sentiment
- Final dataset: 22,000 labeled examples

### Model Development
- Experimented with multiple approaches (BERT, SVM+TF-IDF)
- Selected DistilBERT-base-uncased for final model (40% smaller than BERT, 97% performance)
- Addressed class imbalance (70% positive) with template-based data augmentation
- Fine-tuned with early stopping on F1 score (patience=2)

## Results
- **Final F1 Score:** 85% weighted F1
- Baseline BERT: 71% F1
- SVM + TF-IDF: 55-79% F1 (class-dependent)

## Notebooks
1. **Data Exploration** - Initial dataset analysis, aspect extraction, labeling strategy
2. **Model Comparison** - Benchmarking different approaches (BERT, SVM, etc.)
3. **Data Augmentation** - Addressing class imbalance with synthetic examples
4. **Final Pipeline** - End-to-end training and evaluation

## Technical Stack
- **NLP:** spaCy, Transformers (DistilBERT)
- **Labeling:** OpenAI GPT-3.5 Turbo API
- **Framework:** PyTorch, Hugging Face
- **Evaluation:** Weighted F1, Precision, Recall

## Key Learnings
- Data quality matters more than model size - spent 60% of project time on careful data creation
- Template-based augmentation effectively addresses class imbalance in sentiment tasks
- GPT-3.5 with temperature=0 provides reliable pseudo-labeling for aspect-specific sentiment


*Note: Academic report available in French upon request*
