!pip install tensorflow matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(784, activation="sigmoid"),
    Reshape((28, 28))
])

model.compile(optimizer="adam", loss="mse")
model.fit(x_train, x_train, epochs=10,
          validation_data=(x_test, x_test))

decoded = model.predict(x_test[:10])

plt.figure(figsize=(15, 4))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.axis("off")

    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded[i], cmap="gray")
    plt.axis("off")

plt.show()
