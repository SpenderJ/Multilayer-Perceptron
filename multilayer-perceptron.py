# coding: utf-8

import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        """
        Takes input of data and returns output data

        :param input: data of shape [batch, input_units]
        :return: input
        """
        return input

    def backward(self, input, grad_output):
        """
        Backpropagation on the given input using less gradient
        Because we already received d loss / d layer we only need to multiply it by d layer / d x

        :param input: data of shape [batch, input_units]
        :param grad_output:
        :return: input
        """
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        """
        Apply elementwise ReLU to [batch, input_units] matrix

        :param input: data of shape [batch, input_units]
        :return: input
        """
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output * relu_grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Layer performing a learned affine transformation (f(x) = <W*x> + b)

        :param input_units:
        :param output_units:
        :param learning_rate:
        """

        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Perform an affine transformation (f(x) = <W*x> + b)

        :param input: data of shape [batch, input_units]
        :return: input
        """

        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        """
        Perform (d f / d x = d f / d dense * d dense / d x)
        Then compute gradient w.r.t. weights and biases
        Finally compute gradient weights and biases

        :param input:
        :param grad_output:
        :return:
        """
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
