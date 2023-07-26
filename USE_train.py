from typing import List, Dict
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class USELayer(tf.keras.layers.Layer):
    """
    This class defines a Keras layer that applies the Universal Sentence Encoder (USE) model to generate sentence embeddings.

    Methods:
        call(sentences): Computes sentence embeddings.
    """
    def __init__(self, model_name: str) -> None:
        super(USELayer, self).__init__()
        self.use_model = hub.load(model_name)

    def call(self, sentences):
        # Compute the model output
        embeddings = self.use_model(sentences)

        return embeddings

class USECosineSimilarityModel(tf.keras.Model):
    """
    This class defines a Keras Model for cosine similarity.

    Attributes:
        model (tf.keras.Model): A Keras Model that outputs sentence embeddings.
        loss_metric (tf.keras.metrics.Mean): Metric for tracking the mean loss.

    Methods:
        call(inputs): Computes similarity scores.
        train_step(data): Defines a custom training step.
    """
    def __init__(self, model_name: str):
        super(USECosineSimilarityModel, self).__init__()
        self.model = USELayer(model_name)
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')

    def call(self, inputs):
        input_a = inputs['input_a']
        input_b = inputs['input_b']
        embeddings_a = self.model(input_a)
        embeddings_b = self.model(input_b)
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
if __name__ == '__main__':
    model_name = 'https://tfhub.dev/google/universal-sentence-encoder/4'
    input_a = ['sentence A1', 'sentence A2', 'sentence A3']
    input_b = ['sentence B1', 'sentence B2', 'sentence B3']
    targets = np.array([0.7, 0.8, 0.85])
    data = {
        'input_a': np.array(input_a),
        'input_b': np.array(input_b),
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Initialize your model
    model = USECosineSimilarityModel(model_name)

    # Compile your model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    # Fit the model
    history = model.fit(data, targets, epochs=10, batch_size=32)

    # If you want to validate the model during training, split your data into training and validation sets:
    # history = model.fit(data_train, targets_train, validation_data=(data_val, targets_val), epochs=10, batch_size=32)
