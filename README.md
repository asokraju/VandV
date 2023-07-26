# Verification and Validation (VandV) Project

This repository contains the code for computing similarity scores for system goals. It is a part of the Verification and Validation project.

## Description

This project is part of a Verification and Validation (V&V) process where the goal is to compute similarity scores between different system goals. The goals could be in the form of sentences or short paragraphs. The similarity score is a measure of how similar two goals are in terms of their semantic meaning. This can be useful in various scenarios such as identifying duplicate goals, clustering similar requirements together, or finding related goals in a large dataset.

To compute the similarity scores, we are using a method based on Transformer models, specifically Sentence-BERT (SBERT). SBERT is a modification of the pre-trained BERT network that allows us to derive semantically meaningful sentence embeddings efficiently. These embeddings can then be compared using cosine similarity to provide a measure of how similar two sentences are.

The code is implemented in Python and uses the TensorFlow library for building and training the models, and the Transformers library for accessing pre-trained Transformer models and tokenizers.

## Sentence-BERT (SBERT)

Sentence-BERT (SBERT) is a modification of the BERT model which is specifically optimized for deriving sentence embeddings. In the original BERT model, sentence embeddings were typically derived by taking the output of the first token (the [`CLS`] token) from the last layer of the model. However, it was found that these embeddings were not very effective for semantic textual similarity tasks.

SBERT addresses this issue by adding a pooling operation to the output of BERT to create sentence embeddings. The pooling operation can be mean, max, or CLS token pooling. These sentence embeddings can then be directly used to compute semantic similarity between sentences using cosine similarity.

SBERT has been shown to significantly outperform the original BERT model on various sentence-level tasks like semantic textual similarity, paraphrase identification, and natural language inference. It is also much faster and more efficient than BERT for these tasks because it allows sentence embeddings to be computed in one pass, rather than requiring pairwise comparison of sentences.

In this project, we are using SBERT to compute sentence embeddings for goals and then computing the cosine similarity between these embeddings to get a similarity score.

# Code Overview

The code includes the following classes and functions:

- `TFSTLayer` Class: This class defines a custom Keras layer that applies a Transformer model to generate sentence embeddings. It uses a pre-trained model from the Transformers library, which is specified by the model_name parameter. The call method computes the embeddings, and the mean_pooling method applies mean pooling to the token embeddings.

- `tf_sentence_transformer` Function: This function creates a Keras Model for sentence embeddings. It takes as input the name of the pre-trained Transformer model and the maximum sequence length for tokenization. It returns a Keras Model that outputs sentence embeddings.

- `SBERTCosineSimilarityModel` Class: This class defines a Keras Model for cosine similarity. It uses the tf_sentence_transformer function to create a model that generates sentence embeddings. The call method computes similarity scores between pairs of sentences by first generating embeddings for each sentence and then computing the cosine similarity between these embeddings.

- `tokenize_pairs` Function: This function tokenizes pairs of sentences. It uses a pre-trained BERT tokenizer from the Transformers library to tokenize the sentences. It returns a dictionary that contains the tokenized inputs for each sentence.

Main Section: The main section of the script demonstrates how to use the above classes and functions. It first specifies the name of the pre-trained Transformer model and the maximum sequence length for tokenization. It then tokenizes a pair of sentences and computes the similarity score between them using the SBERTCosineSimilarityModel class.

The code is designed to be modular, with each class and function performing a specific task. This makes it easy to modify or extend the code to suit different goals or to incorporate different Transformer models.

## Dependencies

- Python 3.7+
- TensorFlow 2.3+
- Transformers 4.0+

## Files

- `main.py`: This is the main Python script that contains the code for the models and functions.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/asokraju/VandV.git
    ```


2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python main.py
    ```
