# coding: utf-8

import numpy as np
from softmax import softmax_crossentropy
from softmax import grad_softmax_crossentropy


def train(network, X, y):
    """
    Train our network on a test set X and Y.
    Activate all layers
    Backward prop from last to first so that Dense layers have already one gradient step

    :param network: list of network
    :param X: set of data
    :param y: result matching X
    :return:
    """
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy(logits, y)
    loss_grad = grad_softmax_crossentropy(logits, y)

    # Propagate gradients through the network & Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)  # grad w.r.t. input, also weight updates

    return np.mean(loss)


def predict(network, X):
    """
    Make predictions

    :param network: list of network
    :param X: set of data
    :return: largest prob
    """
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)


def predict_probas(network, X):
    """
    Make predictions

    :param network: list of network
    :param X: set of data
    :return: largest prob
    """
    logits = forward(network, X)[-1]
    return logits


def forward(network, X):
    """
    Activate all networks by applying them sequentially

    :param network: list of network
    :param X: set of data
    :return:
    """
    activations = []
    input = X
    for l in network:
        activations.append(l.forward(input))  # Update to next
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

class Layer:
    """
    Each Layer has be able to perform a pass forward and a pass backward.
    So we have here our main class doing either backward or forward
    """
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


class tanh(Layer):
    """
    Applies non linearity to every element of the network
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        Apply elementwise Hyperbolic tangent function to [batch, input_units] matrix
        :param input: data of shape [batch, input_units]
        :return: input
        """
        relu_forward = np.tanh(input)
        return relu_forward

    def backward(self, input, grad_output):
        # Compute tahn gradient of loss on input
        A = np.tanh(input)
        return grad_output * (1 - np.square(A))


class ReLU(Layer):
    """
    Applies non linearity to every element of the network
    """
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
        # Compute ReLU gradient of loss on input
        relu_grad = input > 0
        return grad_output * relu_grad


class Sigmoid(Layer):
    """
    Applies non linearity to every element of the network
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        Apply elementwise Sigmoid to [batch, input_units] matrix

        :param input: data of shape [batch, input_units]
        :return: input
        """
        relu_forward = 1 / (1 + np.exp(-input))
        return relu_forward

    def backward(self, input, grad_output):
        # Compute sigmoid gradient of loss on input
        A = 1 / (1 + np.exp(-input))
        return grad_output * A * (1 - A)


class Dense(Layer):
    """
    This Layer is an Hidden one. It applies an affine transformation
    """
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
        grad_input = np.dot(grad_output, self.weights.T) # d dense/ d x = weights transposed

        grad_weights = np.dot(input.T, grad_output) # gradient on weights and biases
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        self.weights = self.weights - self.learning_rate * grad_weights # stochastic gradient descent
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
