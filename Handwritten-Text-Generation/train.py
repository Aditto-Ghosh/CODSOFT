import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import LSTM

with open("dataset.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Dataset loaded successfully")
text = text[:200000]

characters = sorted(list(set(text)))

char_to_index = {}
index_to_char = {}

for i, char in enumerate(characters):

    char_to_index[char] = i
    index_to_char[i] = char

print("Total unique characters:", len(characters))

sequence_length = 40

X = []
y = []

for i in range(0, len(text) - sequence_length):

    input_sequence = text[i:i + sequence_length]
    target_character = text[i + sequence_length]

    input_data = []

    for char in input_sequence:
        input_data.append(char_to_index[char])

    X.append(input_data)
    y.append(char_to_index[target_character])

X = np.array(X)
y = np.array(y)

X = X / float(len(characters))
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("Total training samples:", len(X))
model = tf.keras.Sequential()

model.add(
    LSTM(
        256,
        input_shape=(sequence_length, 1)
    )
)

model.add(
    tf.keras.layers.Dense(
        len(characters),
        activation="softmax"
    )
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

print("Training started...\n")

model.fit(
    X,
    y,
    epochs=15,
    batch_size=64
)

print("\nTraining completed")

start = random.randint(0, len(X) - 1)

pattern = list(X[start].flatten())

generated_text = ""

print("\nGenerated Text:\n")

for i in range(500):

    x_input = np.reshape(pattern, (1, len(pattern), 1))

    prediction = model.predict(x_input, verbose=0)

    index = np.argmax(prediction)

    result = index_to_char[index]

    generated_text += result

    pattern.append(index / float(len(characters)))

    pattern = pattern[1:]

print(generated_text)

with open("generated_output.txt", "w", encoding="utf-8") as file:

    file.write(generated_text)

print("\nGenerated text saved successfully")