import sqlite3
import numpy as np

class TextToTextNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, vocab_size, db_url, scale_factor=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.EPSILON = 1e-8
        # Connect to the SQLite database
        self.connection = sqlite3.connect(db_url)
        self.cursor = self.connection.cursor()
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS weights_input_hidden (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weight REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS weights_hidden_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    def save_model(self):
        # Save the current weights to the database
        weights_input_hidden_flat = self.weights_input_hidden.flatten()
        weights_hidden_output_flat = self.weights_hidden_output.flatten()
        self.cursor.execute('DELETE FROM weights_input_hidden')
        self.cursor.executemany(
            'INSERT INTO weights_input_hidden (weight) VALUES (?)',
            [(float(weight),) for weight in weights_input_hidden_flat]
        )
        self.cursor.execute('DELETE FROM weights_hidden_output')
        self.cursor.executemany(
            'INSERT INTO weights_hidden_output (weight) VALUES (?)',
            [(float(weight),) for weight in weights_hidden_output_flat]
        )
        self.connection.commit()
    def load_model(self):
        # Load weights from the database
        self.weights_input_hidden = self.load_weights('weights_input_hidden')
        self.weights_hidden_output = self.load_weights('weights_hidden_output')
    def load_weights(self, table_name):
        self.cursor.execute(f'SELECT weight FROM {table_name}')
        weights = np.array([row[0] for row in self.cursor.fetchall()], dtype=float)
        return weights.reshape(-1, self.hidden_size if table_name == 'weights_input_hidden' else self.vocab_size)
    def softmax(self, x):
        # Softmax activation function with stability trick
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / (exp_x.sum(axis=0, keepdims=True) + self.EPSILON)
    def forward(self, X):
        # Forward propagation through the network
        X = X.astype(float)
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = np.tanh(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.predicted_output = self.softmax(self.output_layer_input)
        return self.predicted_output
    def backward(self, X, y, learning_rate):
        # Backward propagation to update weights
        error = y - self.predicted_output
        output_delta = error
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer_output.T, output_delta) / X.shape[0]
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * (1 - self.hidden_layer_output**2)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta) / X.shape[0]
    def train(self, X, y, epochs, learning_rate, batch_size=32, verbose=True):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                input_sequence = X[i]
                target_sequence = y[i]
                self.predicted_output = np.zeros_like(target_sequence, dtype=float)
                x_one_hot = np.eye(self.vocab_size)[input_sequence]
                y_one_hot = np.eye(self.vocab_size)[target_sequence]
                output = self.forward(x_one_hot)
                self.backward(x_one_hot, y_one_hot, learning_rate)
                loss = -np.sum(y_one_hot * np.log(self.predicted_output + 1e-8))
                total_loss += loss
                if np.isnan(loss) or np.isinf(loss):
                    print("NaN or Inf values encountered. Aborting training.")
                    return
            average_loss = total_loss / len(X)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")
            self.save_model()
    def predict(self, X, temperature=1.0):
        x_one_hot = np.eye(self.vocab_size, dtype=float)[X]
        predictions = self.forward(x_one_hot)
        if predictions.size == 0:
            return None
        scaled_predictions = predictions ** (1 / temperature)
        normalized_predictions = scaled_predictions / (np.sum(scaled_predictions, axis=1, keepdims=True) + self.EPSILON)
        predicted_sequence = np.argmax(normalized_predictions, axis=1)
        return predicted_sequence
    def pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in sequences]
        return np.array(padded_sequences)