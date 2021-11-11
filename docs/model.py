from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub

# This import fixes the following error:
#   FileNotFoundError: Op type not registered 'CaseFoldUTF8' in binary running on DESKTOP-C9UTCQV.
#   Make sure the Op and Kernel are registered in the binary running in this process.
#    Note that if you are loading a saved graph which used ops from tf.contrib,
#   accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph,
#   as contrib ops are lazily registered when the module is first accessed.
import tensorflow_text as text  # noqa

DiscriminatorOutput = Dict[str, tf.Tensor]


class BertEncoder(tf.keras.Model):
    def __init__(self, tfhub_handle_preprocess: str, tfhub_handle_encoder: str):
        super(BertEncoder, self).__init__()
        self.preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name="preprocessing"
        )
        self.encoder = hub.KerasLayer(
            tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )

    @property
    def output_size(self):
        return self.call(["test"]).shape[-1]

    def call(self, inputs, training=False):
        encoder_inputs = self.preprocessing_layer(inputs)
        outputs = self.encoder(encoder_inputs, training=training)
        return outputs["pooled_output"]


class GeneratorModel(tf.keras.Model):
    def __init__(self, hidden_size: int):
        super(GeneratorModel, self).__init__()

        # This is a simplification made by the author.
        output_size = hidden_size
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, name="hidden-layer")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(output_size, name="output-layer")

    def call(self, inputs, training=False) -> tf.Tensor:
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

    def call(self, inputs, training=False) -> DiscriminatorOutput:
        hidden_layer = self.hidden_layer(inputs)
        hidden_layer = tf.nn.leaky_relu(hidden_layer)

        if training:
            hidden_layer = self.dropout(hidden_layer, training=training)

        net = self.output_layer(hidden_layer)
        return {"hidden-layer": hidden_layer, "probabilities": tf.nn.softmax(net)}


class GanBert(tf.keras.Model):
    def __init__(
        self,
        generator: GeneratorModel,
        discriminator: DiscriminatorModel,
        encoder: BertEncoder,
        noise_dim: int,
        batch_size: int,
    ):
        super(GanBert, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.noise_dim = noise_dim
        self.batch_size = batch_size

    @classmethod
    def calculate_feature_loss(
        cls, real_output: DiscriminatorOutput, fake_output: DiscriminatorOutput
    ) -> tf.Tensor:
        generator_features = tf.math.reduce_mean(fake_output["hidden-layer"], axis=0)
        real_features = tf.math.reduce_mean(real_output["hidden-layer"], axis=0)
        return tf.math.reduce_mean(tf.math.square(generator_features - real_features))

    @classmethod
    def calculate_generator_loss(
        cls, real_output, fake_output, epsilon=0.001
    ) -> tf.float32:
        features_loss = cls.calculate_feature_loss(real_output, fake_output)
        generator_loss = -tf.math.reduce_mean(
            1 - tf.math.log(fake_output["probabilities"][:, 0] + epsilon)
        )
        return generator_loss + features_loss

    @classmethod
    def calculate_discriminator_loss(
        cls, target, real_output, fake_output, epsilon=0.001
    ) -> tf.float32:
        loss_supervised = -tf.math.reduce_mean(
            target[:, 1:] * tf.math.log(real_output["probabilities"][:, 1:] + epsilon)
        )
        loss_unsupervised_real = -tf.math.reduce_mean(
            1 - tf.math.log(real_output["probabilities"][:, 0] + epsilon)
        )
        loss_unsupervised_fake = -tf.math.reduce_mean(
            tf.math.log(fake_output["probabilities"][:, 0] + epsilon)
        )
        return loss_supervised + loss_unsupervised_real + loss_unsupervised_fake

    def compile(self, discriminator_optimizer, generator_optimizer):
        super(GanBert, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def train_step(self, data):
        noise = tf.random.normal(
            (self.batch_size, self.noise_dim), mean=0.0, stddev=1.0, dtype=tf.float32
        )
        inputs, target = data
        bert_output = self.encoder(inputs)
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generator_logits = self.generator(noise, training=True)

            real_output = self.discriminator(bert_output, training=True)
            fake_output = self.discriminator(generator_logits, training=True)

            generator_loss = self.calculate_generator_loss(real_output, fake_output)
            discriminator_loss = self.calculate_discriminator_loss(
                target, real_output, fake_output
            )

            gradients_of_generator = generator_tape.gradient(
                generator_loss, self.generator.trainable_variables
            )
            gradients_of_discriminator = discriminator_tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables
            )

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables)
            )
            self.discriminator_optimizer.apply_gradients(
                zip(
                    gradients_of_discriminator,
                    self.discriminator.trainable_variables,
                )
            )

        return {
            "generator-loss": generator_loss,
            "discriminator-loss": discriminator_loss,
        }
