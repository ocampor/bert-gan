# %%

from pathlib import Path

import pandas
import tensorflow as tf
from official.nlp import optimization  # to create AdamW optimizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data import load_datasets
import model

if __name__ == "__main__":
    DATA_PATH = Path("../data")
    data = load_datasets(DATA_PATH)

    X_train = data["labeled.tsv"]
    X_test = data["test.tsv"]
    X_unlabeled = data["unlabeled.tsv"]

    # %%

    encoder = LabelEncoder()
    encoder.fit(X_train.category)

    def transform_category(x: pandas.Series) -> tf.Tensor:
        y_true = encoder.fit_transform(x)
        one_hot = tf.keras.utils.to_categorical(y_true)
        return tf.concat([tf.ones((one_hot.shape[0], 1)), one_hot], axis=1)

    batch_size = 32
    ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_train.message, transform_category(X_train.category))
        )
        .shuffle(buffer_size=1024)
        .batch(batch_size)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices(
            (X_test.message, transform_category(X_test.category))
        )
        .shuffle(buffer_size=1024)
        .batch(batch_size)
    )

    classes = encoder.classes_

    # %%

    bert_encoder = model.BertEncoder(
        tfhub_handle_preprocess="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        tfhub_handle_encoder="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
    )

    discriminator = model.DiscriminatorModel(
        hidden_size=bert_encoder.output_size, output_size=len(classes) + 1
    )

    generator = model.GeneratorModel(hidden_size=bert_encoder.output_size)

    # %%

    epochs = 2
    init_lr = 3e-5
    steps_per_epoch = ds.cardinality().numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    generator_optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    discriminator_optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    gan_bert = model.GanBert(
        generator=generator,
        discriminator=discriminator,
        encoder=bert_encoder,
        noise_dim=100,
        batch_size=32,
    )

    gan_bert.compile(
        discriminator_optimizer=discriminator_optimizer,
        generator_optimizer=generator_optimizer,
    )

    # %%

    history = gan_bert.fit(x=ds, epochs=epochs)

    # %%

    encoder_output = bert_encoder(X_test.message)
    y_pred = discriminator.predict(encoder_output)
    y_pred = tf.argmax(y_pred["probabilities"][:, 1:], axis=-1)
    print(classification_report(X_test.category, encoder.inverse_transform(y_pred)))
