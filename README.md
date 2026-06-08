# Hybrid Semantic Retrieval & Intelligence System (HSRIS)

This repository contains the implementation of a **Hybrid Semantic Retrieval & Intelligence System (HSRIS)**, developed as **Assignment 3** for the **Data Science** course at **FAST NUCES University, Chiniot-Faisalabad (CFD) Campus**.

HSRIS is a Natural Language Processing (NLP) search pipeline engineered **completely from scratch using PyTorch** (avoiding high-level frameworks like scikit-learn for the pipeline components). It is designed to index, clean, embed, and query a dataset of customer support tickets by combining traditional keyword matching with deep semantic representation.

---

## 🛠️ System Architecture & Features

The project is structured into three foundational parts, followed by search pipelines, optimizations, and an interactive UI:

```
                                  [ User Query ]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [ Sparse Keyword Pipeline ]             [ Dense Semantic Pipeline ]
                    │                                       │
            (Custom TF-IDF)                     (TF-IDF Weighted GloVe)
                    │                                       │
             [ TF-IDF Score ]                       [ Semantic Score ]
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                   [ Hybrid Score = α·Sparse + (1-α)·Dense ]
                                        │
                                        ▼
                             [ Top K Search Results ]
```

### 1. Categorical Foundation — The Encoders (From Scratch)
*   **Label Encoding (Priority)**: Maps ordinal ticket priorities (`Low`, `Medium`, `High`, `Critical` to `0`, `1`, `2`, `3`) manually. Includes fallback logic to handle missing or unseen values gracefully with `-1`.
*   **One-Hot Encoding (Channel)**: Converts nominal channel values (`Chat`, `Email`, `Phone`, `Social media`) into sparse binary vectors without using external library encoders. Unseen categories default to an all-zero vector.

### 2. Sparse Representation — Keyword Retrieval
*   **Advanced Text Cleaning**: A pipeline utilizing regex matching to filter out system logs, ALL-CAPS indicators, timestamps, IP addresses, currency formatting, PascalCase/snake_case tokens, and bullet points.
*   **Display Text Re-construction**: Cleans description fields by inserting actual product names into template placeholders and filtering out redundant/generic phrases to keep display text concise and readable.
*   **Custom Tokenizer & Bag of Words**: A regex-based tokenizer that generates word counts, establishing a vocabulary of the top 5,000 most frequent tokens.
*   **N-Gram Generator**: Implements a sliding window mechanism to extract word-level bigrams and trigrams for capturing local phrase structures.
*   **Manual TF-IDF**: Implements Term Frequency (TF) and smoothed Inverse Document Frequency (IDF) matching scikit-learn's smoothing formulation:
    $$\text{IDF}(t) = \log\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$
*   **RAM-Safe Sparse Tensor Storage**: Converts the resulting dense document TF-IDF matrix into a PyTorch sparse tensor representation (`to_sparse()`), reducing GPU/RAM footprint.

### 3. Dense Semantic Layer — Neural Embeddings
*   **GloVe Integration**: Imports pretrained 200-dimensional Global Vectors for Word Representation (GloVe).
*   **Frozen PyTorch Embedding Layer**: Preloads GloVe vectors into `torch.nn.Embedding` with frozen gradients (`requires_grad=False`) for fast neural lookups.
*   **TF-IDF Weighted Sentence Pooling**: Avoids semantic dilution (where common words dilute the meaning of rare technical terms) by calculating sentence embeddings as the TF-IDF weighted average of individual word embeddings:
    $$\vec{v}_{\text{doc}} = \frac{\sum w_i \cdot \vec{v}_i}{\sum w_i}$$
    where $w_i$ is the TF-IDF weight of token $i$, and $\vec{v}_i$ is its GloVe vector.

### 4. Hybrid Search Pipeline
*   **GPU Cosine Similarity**: Uses PyTorch's `F.cosine_similarity` running on the GPU to score queries.
*   **Linear Combination ($a$-blending)**: Integrates keyword and semantic scores to calculate the final ranking metric:
    $$\text{Final Score} = \alpha \cdot \text{Score}_{\text{TF-IDF}} + (1 - \alpha) \cdot \text{Score}_{\text{GloVe}}$$
    *(Default blending parameter $\alpha = 0.4$ as specified in the assignment)*.

### 5. Scale & Performance Optimization
*   **DataParallel & GPU Batching**: Computes similarities using batch matrix multiplication (`torch.mm`) on normalized query/document tensors.
*   **Batch Benchmark**: Benchmarks performance scaling across batch sizes of 10, 25, 50, 75, and 100 queries, profiling execution times.

### 6. Evaluation & User Interface
*   **Precision@5 Evaluation**: Measures performance against ground-truth ticket types.
*   **Side-by-Side Comparison**: Compares retrieval results using keyword-only, semantic-only, and hybrid search.
*   **Interactive Gradio Interface**: A web application that lets users input queries, adjust the $\alpha$ blending weight, and view top retrieved tickets in real time.

---

## 📈 Summary of Components & Pipelines

| Component | Implementation Details |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning Framework** | PyTorch (GPU accelerated) |
| **Preprocessing & Parsing** | Custom Regex Pipeline |
| **Sparse Vector Space** | Custom TF-IDF, Sparse Tensors ($8469 \times 5000$) |
| **Dense Embeddings** | GloVe 200-d Vectors + PyTorch Embedding |
| **UI Framework** | Gradio |

---

## 🚀 Execution & Results
*   **Semantic Search Output**: Successfully identifies semantic relations. For example, a query of `"payment failed"` matches ticket descriptions containing terms like `"Paypal"`, `"Credit card"`, and `"billing error"`.
*   **Performance Scaling**: Demonstrates sub-millisecond execution times when utilizing batched matrix multiplication (`torch.mm`) on CUDA.
