import sqlite3
import numpy as np

class TextToTextNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, vocab_size, db_url, scale_factor=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        # Connect to the MySQL database
        self.connection = sqlite3.connect(
            db_url
        )
        self.cursor = self.connection.cursor()
        # Create tables if they don't existW
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS weights_input_hidden (
                id INT AUTO_INCREMENT PRIMARY KEY,
                weight REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS weights_hidden_output (
                id INT AUTO_INCREMENT PRIMARY KEY,
                weight REAL
            )
        ''')
        if not self.check_table_empty('weights_input_hidden') or not self.check_table_empty('weights_hidden_output'):
            self.load_model()
        else:
            self.weights_input_hidden = scale_factor * np.random.randn(vocab_size, hidden_size).astype(float)
            self.weights_hidden_output = scale_factor * np.random.randn(hidden_size, vocab_size).astype(float)
    def check_table_empty(self, table_name):
        self.cursor.execute(f'SELECT * FROM {table_name}')
        return not self.cursor.fetchone()
    # Modify the save_model method in TextToTextNeuralNetwork class
    def save_model(self):
        # Save the current weights to the database
        weights_input_hidden_flat = self.weights_input_hidden.flatten()
        weights_hidden_output_flat = self.weights_hidden_output.flatten()
        # Insert weights_input_hidden into the database
        self.cursor.execute('DELETE FROM weights_input_hidden')
        self.cursor.executemany('INSERT INTO weights_input_hidden (weight) VALUES (?)', [(float(weight),) for weight in weights_input_hidden_flat])
        # Insert weights_hidden_output into the database
        self.cursor.execute('DELETE FROM weights_hidden_output')
        self.cursor.executemany('INSERT INTO weights_hidden_output (weight) VALUES (?)', [(float(weight),) for weight in weights_hidden_output_flat])
        self.connection.commit()
    def load_model(self):
        # Load weights from the database
        self.cursor.execute('SELECT weight FROM weights_input_hidden')
        weights_input_hidden = np.array([row[0] for row in self.cursor.fetchall()], dtype=float)
        print("Loaded weights_input_hidden shape:", weights_input_hidden.shape)
        self.weights_input_hidden = weights_input_hidden.reshape(self.vocab_size, self.hidden_size)
        self.cursor.execute('SELECT weight FROM weights_hidden_output')
        weights_hidden_output = np.array([row[0] for row in self.cursor.fetchall()], dtype=float)
        print("Loaded weights_hidden_output shape:", weights_hidden_output.shape)
        self.weights_hidden_output = weights_hidden_output.reshape(self.hidden_size, self.vocab_size)
    def softmax(self, x):
        # Softmax activation function with stability trick
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / exp_x.sum(axis=0, keepdims=True)
    def forward(self, X):
        # Forward propagation through the network
        X = X.astype(float)  # Cast X to float
        #print("X shape:", X.shape, "X dtype:", X.dtype)
        #print("weights_input_hidden shape:", self.weights_input_hidden.shape, "weights_input_hidden dtype:", self.weights_input_hidden.dtype)
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        #print("hidden_layer_input shape:", self.hidden_layer_input.shape, "hidden_layer_input dtype:", self.hidden_layer_input.dtype)
        self.hidden_layer_output = np.tanh(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.predicted_output = self.softmax(self.output_layer_input)
        return self.predicted_output
    # Inside the backward method
    def backward(self, X, y, learning_rate):
        # Backward propagation to update weights
        error = y - self.predicted_output
        # Output layer
        output_delta = error
        #print("self.weights_hidden_output shape:", self.weights_hidden_output.shape)
        #print("learning_rate * np.dot(self.hidden_layer_output.T, output_delta) / X.shape[0] shape:",(learning_rate * np.dot(self.hidden_layer_output.T, output_delta) / X.shape[0]).shape)
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer_output.T, output_delta) / X.shape[0]
        # Hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * (1 - self.hidden_layer_output**2)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta) / X.shape[0]  # Transpose X
    def train(self, X, y, epochs, learning_rate, verbose=True):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                input_sequence = X[i]
                target_sequence = y[i]
                # Reset predicted output for each sequence
                self.predicted_output = np.zeros_like(target_sequence, dtype=float)
                # One-hot encode the input and target sequences
                x_one_hot = np.eye(self.vocab_size)[input_sequence]
                y_one_hot = np.eye(self.vocab_size)[target_sequence]
                # Forward and backward pass
                output = self.forward(x_one_hot)
                self.backward(x_one_hot, y_one_hot, learning_rate)
                # Compute loss (cross-entropy)
                loss = -np.sum(y_one_hot * np.log(self.predicted_output + 1e-8))
                total_loss += loss
                # Check for NaN or Inf values in predictions
                if np.isnan(loss) or np.isinf(loss):
                    print("NaN or Inf values encountered. Aborting training.")
                    return
            # Print average loss for the epoch
            average_loss = total_loss / len(X)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")
            # Save the model after each epoch
            self.save_model()
    def predict(self, X, temperature=1.0):
        x_one_hot = np.eye(self.vocab_size, dtype=float)[X]
        predictions = self.forward(x_one_hot)
        scaled_predictions = predictions ** (1 / temperature)
        normalized_predictions = scaled_predictions / np.sum(scaled_predictions, axis=1, keepdims=True)
        predicted_sequence = np.array([np.random.choice(self.vocab_size, p=prob) for prob in normalized_predictions])
        return predicted_sequence
    def pad_sequences(self, sequences):
        # Pad sequences with zeros to have the same length
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = np.zeros((len(sequences), max_len))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences.astype(int)