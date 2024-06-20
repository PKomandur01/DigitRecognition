import random
import math
import json

class OCRNeuralNetwork:
    NN_FILE_PATH = 'neural_network.json'  # Path to save/load neural network weights
    LEARNING_RATE = 0.1  # Learning rate for gradient descent

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, train_indices, use_file=True):
       
        self.num_hidden_nodes = num_hidden_nodes
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        self.train_indices = train_indices
        self.use_file = use_file

        if use_file:
            self._load()  # Load weights from file if use_file is True
        else:
            # Initialize weights randomly if no existing file weights are used
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

    def _rand_initialize_weights(self, size_in, size_out):
        """
        Initialize weights with random values between -0.06 and 0.06.

        Parameters:
        - size_in: Number of input nodes
        - size_out: Number of output nodes

        Returns:
        - List of initialized weights
        """
        return [((x * 0.12) - 0.06) for x in [random.random() for _ in range(size_out * size_in)]]

    def _sigmoid_scalar(self, z):
        """
        Sigmoid activation function for a scalar.

        Parameters:
        - z: Input value

        Returns:
        - Output of the sigmoid function
        """
        return 1 / (1 + math.exp(-z))

    def sigmoid(self, x):
        """
        Sigmoid activation function for a list of values.

        Parameters:
        - x: List of input values

        Returns:
        - List of outputs from the sigmoid function
        """
        return [self._sigmoid_scalar(z) for z in x]

    def sigmoid_prime(self, x):
        """
        Derivative of the sigmoid function.

        Parameters:
        - x: List of input values

        Returns:
        - List of sigmoid derivative values
        """
        return [self.sigmoid(z) * (1 - self.sigmoid(z)) for z in x]

    def train(self):
        """
        Train the neural network using backpropagation.
        """
        for idx in self.train_indices:
            data = {
                'y0': self.data_matrix[idx],
                'label': self.data_labels[idx]
            }

            # Forward propagation
            y1 = [sum([self.theta1[j][i] * data['y0'][i] for i in range(len(data['y0']))]) + self.input_layer_bias[j] for j in range(len(self.theta1))]
            y1 = self.sigmoid(y1)

            y2 = [sum([self.theta2[j][i] * y1[i] for i in range(len(y1))]) + self.hidden_layer_bias[j] for j in range(len(self.theta2))]
            y2 = self.sigmoid(y2)

            # Backpropagation
            actual_vals = [0] * 10
            actual_vals[data['label']] = 1
            output_errors = [actual_vals[i] - y2[i] for i in range(len(actual_vals))]
            hidden_errors = [sum([self.theta2[i][j] * output_errors[j] * self.sigmoid_prime(y1[i]) for j in range(len(output_errors))]) for i in range(len(y1))]

            self.theta1 = [[self.theta1[j][i] + self.LEARNING_RATE * hidden_errors[j] * data['y0'][i] for i in range(len(data['y0']))] for j in range(len(self.theta1))]
            self.theta2 = [[self.theta2[j][i] + self.LEARNING_RATE * output_errors[j] * y1[i] for i in range(len(y1))] for j in range(len(self.theta2))]
            self.hidden_layer_bias = [self.hidden_layer_bias[i] + self.LEARNING_RATE * output_errors[i] for i in range(len(output_errors))]
            self.input_layer_bias = [self.input_layer_bias[i] + self.LEARNING_RATE * hidden_errors[i] for i in range(len(hidden_errors))]

    def predict(self, test):
        y1 = [sum([self.theta1[j][i] * test[i] for i in range(len(test))]) + self.input_layer_bias[j] for j in range(len(self.theta1))]
        y1 = self.sigmoid(y1)

        y2 = [sum([self.theta2[j][i] * y1[i] for i in range(len(y1))]) + self.hidden_layer_bias[j] for j in range(len(self.theta2))]
        y2 = self.sigmoid(y2)

        return y2.index(max(y2))

    def save(self):
      
       # Save the trained neural network weights to a JSON file.
    
        if not self.use_file:
            return

        json_neural_network = {
            "theta1": self.theta1,
            "theta2": self.theta2,
            "b1": self.input_layer_bias,
            "b2": self.hidden_layer_bias
        }
        with open(self.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
       # Load pre-trained neural network weights from a JSON file.
       
        if not self.use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)

        self.theta1 = nn['theta1']
        self.theta2 = nn['theta2']
        self.input_layer_bias = nn['b1']
        self.hidden_layer_bias = nn['b2']

