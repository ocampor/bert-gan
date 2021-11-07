import tensorflow as tf
import tensorflow_hub as hub


class BertClassifier(tf.keras.Model):
    def __init__(self, tfhub_handle_preprocess: str, tfhub_handle_encoder: str, output_size: int):
        super(GeneratorModel, self).__init__()

        self.preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(output_size, activation=None, name='classifier')

    def call(self, inputs, training=False):
        encoder_inputs = self.preprocessing_layer(inputs)
        outputs = self.encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = self.dropout(net)
        net = self.classifier(net)
        return tf.nn.softmax(net)


class GeneratorModel(tf.keras.Model):
    def __init__(self, hidden_size: int):
        super(GeneratorModel, self).__init__()

        # This is a simplifaction made by the author.
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
        net = self.hidden_layer(inputs)
        net = tf.nn.leaky_relu(net)

        if training:
            net = self.self.dropout(net, training=training)

        net = self.output_layer(net)
        return tf.nn.softmax(net)
