# Hybrid Semantic Retrieval & Intelligence System (HSRIS)

This repository contains **Assignment 3** for the **Data Science** course at **FAST NUCES University, Chiniot-Faisalabad (CFD) Campus**.

## Course & Institution Details
* **University:** FAST NUCES University, CFD Campus
* **Course:** Data Science
* **Project Name:** Hybrid Semantic Retrieval & Intelligence System (HSRIS)

## Features & Implementation
The system implements a hybrid search engine for Customer Support Tickets from scratch (without high-level libraries like scikit-learn for ML pipelines):
1. **Categorical Encoders**: Custom Label Encoding and One-Hot Encoding.
2. **Sparse Retrieval (Keyword)**: Preprocessing, Text Cleaning, Custom Tokenizer, Bag of Words, N-Gram Generator (Bigrams & Trigrams), and custom TF-IDF implementation stored in a RAM-safe Sparse Tensor.
3. **Dense Semantic Layer**: Pretrained GloVe embeddings and TF-IDF weighted sentence embedding.
4. **Hybrid Search Pipeline**: Combined keyword and semantic search.
5. **Performance Optimization**: Dual GPU batch similarity computation.
6. **Visualization & Evaluation**: Precision@5 evaluation, side-by-side comparison, and qualitative examples.
7. **Gradio App**: Interactive web interface for testing the search engine.
