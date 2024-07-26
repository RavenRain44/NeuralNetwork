import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, input_data) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return self.activation(input_data)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(self.activation_prime(self.input), output_gradient)


class Tanh(Activation):
    def __init__(self):
        super().__init__(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2)
