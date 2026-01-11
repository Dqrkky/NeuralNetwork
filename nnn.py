import sqlite3
import math
import random


class TextToTextNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, vocab_size, db_url, scale_factor=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.EPSILON = 1e-8

        self.connection = sqlite3.connect(db_url)
        self.cursor = self.connection.cursor()

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
            self.weights_input_hidden = self.random_matrix(self.vocab_size, self.hidden_size, scale_factor)
            self.weights_hidden_output = self.random_matrix(self.hidden_size, self.vocab_size, scale_factor)

    def random_matrix(self, rows, cols, scale):
        return [[random.gauss(0, 1) * scale for _ in range(cols)] for _ in range(rows)]

    def check_table_empty(self, table_name):
        self.cursor.execute(f'SELECT * FROM {table_name}')
        return not self.cursor.fetchone()

    def save_model(self):
        self.cursor.execute('DELETE FROM weights_input_hidden')
        self.cursor.executemany(
            'INSERT INTO weights_input_hidden (weight) VALUES (?)',
            [(w,) for row in self.weights_input_hidden for w in row]
        )

        self.cursor.execute('DELETE FROM weights_hidden_output')
        self.cursor.executemany(
            'INSERT INTO weights_hidden_output (weight) VALUES (?)',
            [(w,) for row in self.weights_hidden_output for w in row]
        )
        self.connection.commit()

    def load_model(self):
        self.weights_input_hidden = self.load_weights('weights_input_hidden', self.vocab_size, self.hidden_size)
        self.weights_hidden_output = self.load_weights('weights_hidden_output', self.hidden_size, self.vocab_size)

    def load_weights(self, table_name, rows, cols):
        self.cursor.execute(f'SELECT weight FROM {table_name}')
        weights = [row[0] for row in self.cursor.fetchall()]
        return [weights[i * cols:(i + 1) * cols] for i in range(rows)]

    def softmax(self, x):
        max_x = max(x)
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp = sum(exp_x) + self.EPSILON
        return [i / sum_exp for i in exp_x]

    def dot(self, a, b):
        return [sum(a[i] * b[i][j] for i in range(len(a))) for j in range(len(b[0]))]

    def matmul(self, A, B):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def transpose(self, M):
        return list(map(list, zip(*M)))

    def tanh(self, x):
        return [math.tanh(i) for i in x]

    def forward(self, X):
        self.hidden_layer_input = self.matmul(X, self.weights_input_hidden)
        self.hidden_layer_output = [self.tanh(row) for row in self.hidden_layer_input]
        self.output_layer_input = self.matmul(self.hidden_layer_output, self.weights_hidden_output)
        self.predicted_output = [self.softmax(row) for row in self.output_layer_input]
        return self.predicted_output

    def backward(self, X, y, learning_rate):
        output_delta = [[y[i][j] - self.predicted_output[i][j] for j in range(len(y[0]))] for i in range(len(y))]
        trans_hidden = self.transpose(self.hidden_layer_output)
        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[0])):
                gradient = sum(trans_hidden[i][k] * output_delta[k][j] for k in range(len(output_delta)))
                self.weights_hidden_output[i][j] += learning_rate * gradient / len(X)

        hidden_error = self.matmul(output_delta, self.transpose(self.weights_hidden_output))
        hidden_delta = [[hidden_error[i][j] * (1 - self.hidden_layer_output[i][j] ** 2)
                         for j in range(len(hidden_error[0]))] for i in range(len(hidden_error))]

        trans_input = self.transpose(X)
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[0])):
                gradient = sum(trans_input[i][k] * hidden_delta[k][j] for k in range(len(hidden_delta)))
                self.weights_input_hidden[i][j] += learning_rate * gradient / len(X)

    def train(self, X, y, epochs, learning_rate, batch_size=32, verbose=True):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                for j in range(len(batch_X)):
                    input_sequence = batch_X[j]
                    target_sequence = batch_y[j]
                    x_one_hot = self.one_hot_sequence(input_sequence)
                    y_one_hot = self.one_hot_sequence(target_sequence)
                    self.forward(x_one_hot)
                    self.backward(x_one_hot, y_one_hot, learning_rate)
                    loss = sum(
                        -target * math.log(pred + self.EPSILON)
                        for target_row, pred_row in zip(y_one_hot, self.predicted_output)
                        for target, pred in zip(target_row, pred_row)
                    )
                    if math.isnan(loss) or math.isinf(loss):
                        print("NaN or Inf values encountered. Aborting training.")
                        return
                    total_loss += loss
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(X)}")
            self.save_model()

    def predict(self, X, temperature=1.0):
        x_one_hot = self.one_hot_sequence(X)
        predictions = self.forward(x_one_hot)
        scaled_predictions = [
            [p ** (1 / temperature) for p in row] for row in predictions
        ]
        normalized_predictions = [
            [p / (sum(row) + self.EPSILON) for p in row] for row in scaled_predictions
        ]
        predicted_sequence = [row.index(max(row)) for row in normalized_predictions]
        return predicted_sequence

    def one_hot_sequence(self, seq):
        return [[1.0 if i == idx else 0.0 for i in range(self.vocab_size)] for idx in seq]

    def pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]
        return padded_sequences
