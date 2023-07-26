# Verification and Validation (VandV) Project

This repository includes the code required to evaluate similarity scores among system objectives. The endeavor forms a component of the Verification and Validation process.

## Description

The initiative is an integral part of a Verification and Validation (V&V) scheme where the objective is to evaluate semantic similarity scores between distinct system goals. The goals might take the form of sentences or short paragraphs. The similarity score signifies how analogous two goals are with regards to their semantic meaning. It proves useful in various scenarios like detecting duplicate goals, clustering analogous goals, or pinpointing related goals within an extensive dataset.

For similarity score calculation, we utilize a method centered around Transformer models, specifically Universal Sentence Encoder (USE). USE is a model developed by Google that generates semantically meaningful sentence embeddings efficiently. These embeddings can subsequently be compared employing cosine similarity to provide a measure of semantic similarity between two sentences.

The code is implemented in Python, employing TensorFlow for model building and training, and the Transformers library for accessing pre-trained Transformer models and tokenizers.

## Universal Sentence Encoder (USE)

The Universal Sentence Encoder (USE) is a model developed by Google that provides high-quality sentence embeddings. Unlike the original BERT model where sentence embeddings are usually derived by taking the output of the first token (the [`CLS`] token) from the final model layer, USE is optimized for generating meaningful and efficient sentence embeddings.

USE tackles this issue by applying a transformation on the output of the Transformer to generate sentence embeddings. The resulting sentence embeddings can be utilized directly to compute semantic similarity between sentences using cosine similarity.

USE has demonstrated substantial performance on various sentence-level tasks such as semantic textual similarity, paraphrase identification, and natural language inference. It is much faster and more efficient than BERT for these tasks as it allows the calculation of sentence embeddings in a single pass, avoiding the need for pairwise comparison of sentences.

In this project, we utilize USE to compute sentence embeddings for the objectives, and then compute the cosine similarity between these embeddings to derive a similarity score.

# Code Overview

The code includes the following classes and functions:

- `USEEmbeddingLayer Class: This class defines a custom Keras layer that applies the Universal Sentence Encoder model to generate sentence embeddings. It uses a pre-trained model from the TensorFlow Hub, specified by the model_url parameter. The call method computes the embeddings for given inputs.

- `USECosineSimilarityModel Class: This class defines a Keras Model for cosine similarity. It uses the USEEmbeddingLayer to create a model that generates sentence embeddings. The call method computes similarity scores between pairs of sentences by first generating embeddings for each sentence and then computing the cosine similarity between these embeddings.

- `Main Section`: The main section of the script demonstrates how to use the above classes. It first specifies the URL of the pre-trained Universal Sentence Encoder model. It then defines pairs of sentences for which to compute similarity scores. The USECosineSimilarityModel is instantiated with the specified model URL, compiled with an optimizer, and then trained using the provided pairs of sentences and their corresponding target similarity scores.

This code is designed to be modular, with each class performing a specific task. This makes it easy to modify or extend the code to suit different goals or to incorporate different sentence encoding models. For example, you could replace the Universal Sentence Encoder with another sentence encoder by creating a new embedding layer class that applies the alternative sentence encoder.

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
