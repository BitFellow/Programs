import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(96, 96),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=32,
          validation_data=(x_test, y_test))

print("Test Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

predictions = np.argmax(model.predict(x_test[:25], verbose=0), axis=1)

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis("off")
    color = "green" if predictions[i] == y_test[i][0] else "red"
    plt.xlabel(class_names[predictions[i]], color=color)

plt.tight_layout()
plt.show()