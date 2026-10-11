import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense

data = pd.DataFrame({
    "text": [
        "This document talks about sports",
        "This is about politics",
        "This article is about technology"
    ],
    "label": [0, 1, 2]
})

texts = data["text"].tolist()
labels = np.array(data["label"])
num_classes = len(set(labels))

tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=150)

x_train, x_test = X[:2], X[2:]
y_train, y_test = labels[:2], labels[2:]

model = Sequential([
    Embedding(10000, 128),
    Bidirectional(LSTM(64)),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, validation_split=0.5)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

text = ["This document talks about technology and AI."]
seq = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=150)
pred = model.predict(seq)

print(text[0])
print("Predicted Class:", np.argmax(pred))
