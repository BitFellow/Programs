!pip install -U tensorflow-datasets
!pip install importlib-resources

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

train, test = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True)

model = models.Sequential([
    layers.TextVectorization(max_tokens=10000, output_sequence_length=200),
    layers.Embedding(10000, 64, mask_zero=True),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.layers[0].adapt(train.map(lambda x, y: x))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train.batch(512), epochs=5, validation_data=test.batch(512))

print("Test Accuracy:", model.evaluate(test.batch(512))[1])

reviews = [
    "The movie was fantastic and I loved it",
    "The movie was terrible and boring",
    "The film was wonderful",
    "The acting was bad"
]

predictions = model.predict(tf.constant(reviews))

for review, score in zip(reviews, predictions):
    print(review, "->", "Positive" if score[0] > 0.5 else "Negative")
