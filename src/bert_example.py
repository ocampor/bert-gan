from pathlib import Path

import pandas
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Required by bert_preprocess_model
from official.nlp import optimization  # to create AdamW optimizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data import load_datasets

if __name__ == "__main__":
    DATA_PATH = Path("data")
    data = load_datasets(DATA_PATH)

    X_train = data["labeled.tsv"]
    X_test = data["test.tsv"]
    X_unlabeled = data["unlabeled.tsv"]

    encoder = LabelEncoder()
    encoder.fit(X_train.category)

    def transform_category(x: pandas.Series) -> tf.Tensor:
        y_true = encoder.fit_transform(x)
        return tf.keras.utils.to_categorical(y_true)

    batch_size = 32
    ds = tf.data.Dataset.from_tensor_slices(
        (X_train.message, transform_category(X_train.category))
    ).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices(
        (X_test.message, transform_category(X_test.category))
    ).batch(batch_size)

    classes = encoder.classes_

    # %%

    tfhub_handle_preprocess = (
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

    tfhub_handle_encoder = (
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    )
    bert_model = hub.KerasLayer(tfhub_handle_encoder)

    # %%

    def classifier_factory():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name="preprocessing"
        )
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(len(classes), activation=None, name="classifier")(
            net
        )
        net = tf.nn.softmax(net)
        return tf.keras.Model(text_input, net)

    # %%

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalAccuracy()

    # %%

    epochs = 100
    steps_per_epoch = ds.cardinality().numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    # %%

    classifier_model = classifier_factory()
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # %%

    print(f"Training model with {tfhub_handle_encoder}")
    history = classifier_model.fit(x=ds, epochs=epochs)

    # %%

    y_pred = classifier_model.predict(ds_val)
    y_pred = tf.argmax(y_pred, axis=-1)
    print(classification_report(X_test.category, encoder.inverse_transform(y_pred)))
