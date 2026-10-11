import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = np.array([120, 150, 160, 200, 220, 260, 300,
                 340, 380, 420, 460, 500, 540])

window = 5
X, y = [], []

for i in range(len(data) - window):
    X.append(data[i:i + window])
    y.append(data[i + window])

X = np.array(X).reshape(-1, window, 1)
y = np.array(y)

model = Sequential([
    LSTM(64, activation="relu", input_shape=(window, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
history = model.fit(X, y, epochs=200, verbose=1)

last_values = data[-window:].reshape(1, window, 1)
next_value = model.predict(last_values, verbose=0)

print("Next predicted value:", float(next_value[0, 0]))

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
