import nnn as nn # your custom neural network module

# Training Data
X_train_text = ["cat", "dog", "apple", "banana"]
y_train_text = ["animal", "animal", "fruit", "fruit"]

# Char to index and vice versa
char_to_index = {char: idx for idx, char in enumerate(set("".join(X_train_text + y_train_text)))}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Encoding input and target sequences (pure Python lists)
X_train_indices = [[char_to_index[char] for char in word] for word in X_train_text]
y_train_indices = [[char_to_index[char] for char in word] for word in y_train_text]

# Initialize the neural network
text_to_text_nn = nn.TextToTextNeuralNetwork(
    input_size=max(len(seq) for seq in X_train_indices),
    hidden_size=16,
    output_size=max(len(seq) for seq in y_train_indices),
    vocab_size=len(char_to_index),
    db_url="model.sql"
)

# Padding sequences to ensure uniform length
X_train_indices = text_to_text_nn.pad_sequences(sequences=X_train_indices)
y_train_indices = text_to_text_nn.pad_sequences(sequences=y_train_indices)

# Train the model
text_to_text_nn.train(
    X=X_train_indices,
    y=y_train_indices,
    epochs=100,
    learning_rate=0.5,
    batch_size=16,
    verbose=True
)

# Test Data
X_test_text = ["cat", "dog", "apple", "banana"]
X_test_indices = [[char_to_index[char] for char in word] for word in X_test_text]

# Padding test sequences
X_test_indices = text_to_text_nn.pad_sequences(X_test_indices)

# Predict and print output
print("-" * 20 + "\n")
for i in range(len(X_test_indices)):
    predicted_sequence = text_to_text_nn.predict(X_test_indices[i], temperature=0.5)
    if predicted_sequence is None:
        print(f"Input: {X_test_text[i]}, Predicted: <empty sequence>")
        continue
    valid_indices = [index for index in predicted_sequence if 0 <= index < len(index_to_char)]
    predicted_words = [index_to_char.get(index, '<unknown>') for index in valid_indices]
    output = (
        f"Input: {X_test_text[i]}" + "\n" +
        f"Predicted: {''.join(predicted_words)}" + "\n" +
        f"Predicted Sequence: {valid_indices}" + "\n" +
        f"X_test_indices: {X_test_indices[i]}" + "\n"
    )
    print(output)
print("-" * 20 + "\n")
