import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential([
    Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    Conv2D(32, kernel_size=3, activation="relu"),
    Flatten(),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=3
)

score = model.evaluate(X_test, Y_test, verbose=0)

sample = X_test[4:5]
prediction = model.predict(sample)

plt.imshow(sample.reshape(28, 28), cmap="gray")
plt.show()

classes = np.argmax(prediction, axis=1)

print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")
print("Output Class:", classes)
