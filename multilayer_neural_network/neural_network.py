import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def forward(self, inputs):
        # Calculate the output of the hidden layers
        self.hidden_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate the final output
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def train(self, inputs, labels, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            self.forward(inputs)

            # Calculate the error
            error = labels - self.final_output

            # Backpropagation
            output_error = error * sigmoid_derivative(self.final_output)
            hidden_layer_error = output_error.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)

            # Update weights
            self.weights_hidden_output += self.hidden_output.T.dot(output_error) * learning_rate
            self.weights_input_hidden += inputs.T.dot(hidden_layer_error) * learning_rate

            # Display the current error every 1000 epochs
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Error {np.mean(np.abs(error))}')

# Input data (2 inputs)
input_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Corresponding labels (binary classifier, you can use 0, 1 or -1, +1)
labels = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize the neural network
neural_network = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the neural network
neural_network.train(input_data, labels, learning_rate=0.1, epochs=10000)

# Test the trained network
predictions = neural_network.forward(input_data)
print("\nPredictions:")
print(predictions)
