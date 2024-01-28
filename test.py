import nn
import numpy as np
import credentials

X_train_text = ["cat", "dog", "apple", "banana"]
y_train_text = ["animal", "animal", "fruit", "fruit"]

char_to_index = {char: idx for idx, char in enumerate(set("".join(X_train_text + y_train_text)))}
X_train_indices = [np.array([char_to_index[char] for char in word]) for word in X_train_text]
y_train_indices = [np.array([char_to_index[char] for char in word]) for word in y_train_text]

text_to_text_nn = nn.TextToTextNeuralNetwork(
    input_size=max(map(len, X_train_indices)),
    hidden_size=16,
    output_size=max(map(len, y_train_indices)),
    vocab_size=len(char_to_index),
    db_url="model.sql"
)

X_train_indices = text_to_text_nn.pad_sequences(X_train_indices)
y_train_indices = text_to_text_nn.pad_sequences(y_train_indices)

text_to_text_nn.train(X_train_indices, y_train_indices, epochs=100, learning_rate=0.01, batch_size=32, verbose=True)

X_test_text = ["cat", "dog", "apple", "banana"]
X_test_indices = [np.array([char_to_index[char] for char in word]) for word in X_test_text]

X_test_indices = text_to_text_nn.pad_sequences(X_test_indices)

index_to_char = {idx: char for char, idx in char_to_index.items()}

for i in range(len(X_test_indices)):
    input_sequence = X_test_indices[i]
    predicted_sequence = text_to_text_nn.predict(input_sequence, temperature=0.5)
    if predicted_sequence is None:
        print(f"Input: {X_test_text[i]}, Predicted: <empty sequence>")
        continue
    valid_indices = [index for index in predicted_sequence if 0 <= index < len(index_to_char)]
    predicted_words = [index_to_char.get(index, '<unknown>') for index in valid_indices]
    print(f"Input: {X_test_text[i]}, Predicted: {''.join(predicted_words)}, Predicted Sequence: {valid_indices}")