from typing import List, Dict, Tuple
import tensorflow as tf
from transformers import TFAutoModel, BertTokenizer
import numpy as np

class TFSTLayer(tf.keras.layers.Layer):
    """
    This class defines a Keras layer that applies a Transformer model to generate sentence embeddings.

    Attributes:
        tf_model (TFAutoModel): Pretrained Transformer model.

    Methods:
        call(input_ids, attention_mask, token_type_ids, normalize=True): Computes sentence embeddings.
        mean_pooling(token_embeddings, attention_mask): Applies mean pooling to token embeddings.
    """
    def __init__(self, model_name: str) -> None:
        super(TFSTLayer, self).__init__()
        self.tf_model = TFAutoModel.from_pretrained(model_name)

    def call(self, input_ids, attention_mask, token_type_ids, normalize=True):
        # Compute the model output
        output = self.tf_model(input_ids, attention_mask, token_type_ids)

        # Compute the token embeddings
        token_embeddings = output.last_hidden_state  # shape=(B, max_seq_length, n_embd), dtype=float32

        # Mean Pooling
        embedding = self.mean_pooling(token_embeddings, attention_mask)  # shape=(B, n_embd), dtype=float32

        if normalize:
            embedding, _ = tf.linalg.normalize(embedding, 2, axis=1)  # shape=(B, n_embd), dtype=float32

        return embedding

    def mean_pooling(self, token_embeddings, attention_mask):
        attention_mask = tf.expand_dims(attention_mask, axis=-1)  # shape=(B, max_seq_length, 1), dtype=int32
        attention_mask = tf.broadcast_to(attention_mask, tf.shape(token_embeddings))  # shape=(B, max_seq_length, n_embd), dtype=int32
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)  # shape=(B, max_seq_length, n_embd), dtype=float32
        token_embeddings = token_embeddings * attention_mask  # shape=(B, max_seq_length, n_embd), dtype=float32

        # Taking mean over all the tokens (max_seq_length axis)
        mean_embeddings = tf.reduce_sum(token_embeddings, axis=1)  # shape=(B, n_embd), dtype=float32
        # Alternatively, you can replace the `mean_pooling` method with `tf.keras.layers.GlobalAveragePooling1D`:
        # mean_pooling = tf.keras.layers.GlobalAveragePooling1D()
        # mean_embeddings = mean_pooling(token_embeddings)
        return mean_embeddings

def tf_sentence_transformer(model_name:str, max_seq_length) -> tf.keras.Model:
    """
    This function creates a Keras Model for sentence embeddings.

    Args:
        model_name (str): The name of the pretrained Transformer model.
        max_seq_length (int): The maximum sequence length for tokenization.

    Returns:
        model (tf.keras.Model): A Keras Model that outputs sentence embeddings.
    """
    input_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    token_type_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    tfst_layer = TFSTLayer(model_name)
    output = tfst_layer(input_ids, attention_mask, token_type_ids)
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    return model

class SBERTCosineSimilarityModel(tf.keras.Model):
    """
    This class defines a Keras Model for cosine similarity.

    Attributes:
        tokenizer (BertTokenizerFast): Pretrained BERT tokenizer.
        model (tf.keras.Model): A Keras Model that outputs sentence embeddings.
        loss_metric (tf.keras.metrics.Mean): Metric for tracking the mean loss.

    Methods:
        call(inputs): Computes similarity scores.
        train_step(data): Defines a custom training step.
    """
    def __init__(self, model_name: str, max_seq_length: int):
        super(SBERTCosineSimilarityModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = tf_sentence_transformer(model_name, max_seq_length)
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')

    def call(self, inputs):
        input_ids_a = inputs['input_ids_a']
        input_ids_b = inputs['input_ids_b']
        attention_mask_a = inputs['attention_mask_a']
        attention_mask_b = inputs['attention_mask_b']
        token_type_ids_a = inputs['token_type_ids_a']
        token_type_ids_b = inputs['token_type_ids_b']
        embeddings_a = self.model([input_ids_a, attention_mask_a, token_type_ids_a])
        embeddings_b = self.model([input_ids_b, attention_mask_b, token_type_ids_b])
        normalized_a = tf.nn.l2_normalize(embeddings_a, axis=1)
        normalized_b = tf.nn.l2_normalize(embeddings_b, axis=1)
        similarity_scores = tf.reduce_sum(tf.multiply(normalized_a, normalized_b), axis=1)
        return similarity_scores

    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            similarity_scores = self(inputs)
            loss = tf.keras.losses.MSE(targets, similarity_scores)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

def tokenize_pairs(input_a:List[str], input_b:List[str], tokenizer:BertTokenizer, max_length:int) -> Dict[str, np.array]:
    """
    This function tokenizes pairs of sentences.

    Args:
        input_a (List[str]): The first list of sentences.
        input_b (List[str]): The second list of sentences.
        tokenizer (BertTokenizer): Pretrained BERT tokenizer.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        data (dict): A dictionary that contains tokenized inputs for each sentence.
    """
    tokenized_sen_a = tokenizer(input_a, padding='max_length', max_length=max_length, truncation=True)
    tokenized_sen_b = tokenizer(input_b, padding='max_length', max_length=max_length, truncation=True)
    return {
        'input_ids_a': np.array(tokenized_sen_a['input_ids']),
        'input_ids_b': np.array(tokenized_sen_b['input_ids']),
        'attention_mask_a': np.array(tokenized_sen_a['attention_mask']),
        'attention_mask_b': np.array(tokenized_sen_b['attention_mask']),
        'token_type_ids_a': np.array(tokenized_sen_a['token_type_ids']),
        'token_type_ids_b': np.array(tokenized_sen_b['token_type_ids']),
    }

if __name__ == '__main__':


    # load data 
    #data = np.load('filename.npy')

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    max_seq_length = 512
    tokenizer = BertTokenizer.from_pretrained(model_name)
    input_a = ['sentence A1', 'sentence A2', 'sentence A3']
    input_b = ['sentence B1', 'sentence B2', 'sentence B3']
    targets = np.array([0.7, 0.8, 0.85])
    # Tokenize the input data
    data = tokenize_pairs(input_a, input_b, tokenizer, max_length=max_seq_length)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Initialize your model
    model = SBERTCosineSimilarityModel(model_name, max_seq_length)

    # Compile your model
    model.compile(optimizer=optimizer)