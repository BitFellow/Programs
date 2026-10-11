import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y

print(df.head())
print(df.iloc[50:100])

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(10, input_shape=(4,), activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

plt.figure(figsize=(12, 5))

for j, labels in enumerate([y_true, y_pred]):
    plt.subplot(1, 2, j + 1)

    for i in range(3):
        idx = labels == i
        plt.scatter(
            X_test[idx, 2],
            X_test[idx, 3],
            label=iris.target_names[i]
        )

    plt.title("True Labels" if j == 0 else "Predicted Labels")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()

plt.tight_layout()
plt.show()
