import tensorflow as tf
import tensorflow_hub as hub

# This import fixes the following error:
#   FileNotFoundError: Op type not registered 'CaseFoldUTF8' in binary running on DESKTOP-C9UTCQV.
#   Make sure the Op and Kernel are registered in the binary running in this process.
#    Note that if you are loading a saved graph which used ops from tf.contrib,
#   accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph,
#   as contrib ops are lazily registered when the module is first accessed.
import tensorflow_text as text  # noqa


class BertEncoder(tf.keras.Model):
    def __init__(self, tfhub_handle_preprocess: str, tfhub_handle_encoder: str):
        super(BertEncoder, self).__init__()
        self.preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')

    def call(self, inputs, training=False):
        encoder_inputs = self.preprocessing_layer(inputs)
        outputs = self.encoder(encoder_inputs)
        return outputs['pooled_output']


class GeneratorModel(tf.keras.Model):
    def __init__(self, hidden_size: int):
        super(GeneratorModel, self).__init__()

        # This is a simplification made by the author.
        output_size = hidden_size
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, name="hidden-layer")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(output_size, name="output-layer")

    def call(self, inputs, training=False):
        net = self.hidden_layer(inputs)
        net = tf.nn.leaky_relu(net)

        if training:
            net = self.dropout(net, training=training)

        return self.output_layer(net)


class DiscriminatorModel(tf.keras.Model):
    def __init__(self, hidden_size: int, output_size: int):
        super(DiscriminatorModel, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, name="hidden-layer")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(output_size, name="output-layer")

    def call(self, inputs, training=False):
        hidden_layer = self.hidden_layer(inputs)
        hidden_layer = tf.nn.leaky_relu(hidden_layer)

        if training:
            hidden_layer = self.self.dropout(hidden_layer, training=training)

        net = self.output_layer(hidden_layer)
        return {
            "hidden-layer": hidden_layer,
            "probabilities": tf.nn.softmax(net)
        }
