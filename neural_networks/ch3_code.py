import sys
sys.stdout.flush()
import os
import io
import gzip
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import matplotlib.patches as patches
import json
import time


def sigmoid(z):
    """The sigmoid function, JIT-compiled for speed."""
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    """The ReLU function, JIT-compiled for speed."""
    return np.maximum(0, z)

def sigmoid_prime(z):
    """Derivative of the sigmoid function, JIT-compiled for speed."""
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1 - s)

def relu_prime(z):
    """Derivative of the ReLU function, JIT-compiled for speed."""
    return np.where(z > 0, 1.0, 0.0)

class NN2:
    def __init__(self, layer_sizes, rand_seed=0, activation="relu", cost="mse"):
        """Initialize the neural network with the given layer sizes.
        For example, if layer_sizes = [50, 15, 20, 10], then we have a
        network with 50 input neurons, 15 neurons in hidden layer 0,
        20 neurons in hidden layer 1, and 10 output neurons (layer 3).
        """
        self.number_of_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.rng = np.random.default_rng(seed=rand_seed)
        if activation == 'relu':
            self.activation_func = relu
            self.activation_prime = relu_prime
        elif activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_prime = sigmoid_prime
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'sigmoid'.")
        if cost == 'mse':
            self.cost_func = lambda output, target: np.mean((output - target) ** 2) / 2
            self.cost_derivative = lambda output, target: (output - target)
        if cost == 'cross-entropy':
            self.cost_func = lambda output

        else:
            raise ValueError("Unsupported cost function. Use 'mse'.")
        # randomly initialize weights and biases
        # input layer has no weights nor biases, so we skip it.
        rng = np.random.default_rng(seed=rand_seed)
        # each neuron get 1 bias, so bias vector has the size of the layer
        # we skip the first (input) layer, it doesn't have biases
        # I made the biases to be matrices of shape (N_b, 1) instead of vectors of shape (N_b,)
        # to make the broadcasting work more smoothly in the feedforward and backpropagation functions.
        self.biases = [rng.normal(loc=0, scale=1, size=(N_b, 1))
                       for N_b in layer_sizes[1:]]
        # each neuron in layer Right is connected to all neurons in layer Left,
        # so weight matrix has the shape (size_right, size_left)
        # again, we skip the first (input) layer, it doesn't have weights
        # scale_for_weights = np.sqrt(2.0/size_left)
        scale_for_weights = 1
        self.weights = [rng.normal(loc=0, scale=scale_for_weights, size=(size_right, size_left))
                        for size_left, size_right in
                          zip(layer_sizes[:-1],layer_sizes[1:])
                        ]
    
    def feedforward(self, a):
        """given input `a` from the first layer,
           we sequencially compute the activations of each layer
           `feedforward` returns the activations of last (output) layer
        """
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.activation_func(z)
        return a
    
    def stochastic_gradient_descent(self, training_data, test_data, epochs, batch_size, eta, monitoring=False):
        n = len(training_data)
        try:
            if monitoring:
                monitoring_dict = {
                    "epoch": [],
                    "runtime_seconds": [],
                    "test_cost": [],
                    "test_accuracy": [],
                    "train_cost": [],
                    "train_accuracy": []
                }
                x_train = np.hstack([x for x, y in training_data])
                y_train = np.hstack([y for x, y in training_data])
                x_test = np.hstack([x for x, y in test_data])
                y_test = np.hstack([y for x, y in test_data])
            pbar = tqdm(
                range(1, epochs + 1),
                total=epochs,
                # bar_format="{desc} {bar} {postfix}",
                bar_format="{desc} {bar} ETA:{remaining} {postfix}", 
                leave=True,
                mininterval=0.0,   # update every epoch
                maxinterval=0.0
            )
            pbar.container.children[1].layout.width = "150px"
            for epoch_j in pbar:
                # shuffle training data at the beginning of each epoch
                self.rng.shuffle(training_data)
                # split training data into batches
                batches = [
                    training_data[k:k+batch_size]
                    for k in range(0, n, batch_size)
                ]

                start_time = time.perf_counter()
                # now loop over batches, update weights
                for batch in batches:
                    self.update_params_batch(batch, eta)
                duration_seconds = time.perf_counter() - start_time
                accuracy = self.evaluate(test_data) / len(test_data)
                pbar.set_description(f"Epoch: {epoch_j}/{epochs}")
                pbar.set_postfix_str(
                    f"test acc = {accuracy:.2%}, time/epoch = {duration_seconds:.2f}s"
                )
                if monitoring:
                    monitoring_dict["epoch"].append(epoch_j)
                    monitoring_dict["runtime_seconds"].append(duration_seconds)
                    monitoring_dict["train_accuracy"].append(self.evaluate(training_data)/len(training_data))
                    monitoring_dict["test_accuracy"].append(accuracy)
                    monitoring_dict["train_cost"].append(self.cost_func(self.feedforward(x_train), y_train))
                    monitoring_dict["test_cost"].append(self.cost_func(self.feedforward(x_test), y_test))
            if monitoring:
                return monitoring_dict
        except KeyboardInterrupt:
            if 'pbar' in locals():
                pbar.close()
            print("\n\nTraining interrupted by user. Weights preserved.")
    
    def update_params_batch(self, batch, eta):
        # 1. make input matrix, use column-major order, so each column is a training example, and each row is a feature.
        # input = np.array([data[0] for data in batch]).T
        # 2. make label matrix, use column-major order, so each column is a training example, and each row is a label.
        # target = np.array([data[1] for data in batch]).T
        input = np.hstack([data[0].reshape(-1, 1) for data in batch])
        target = np.hstack([data[1].reshape(-1, 1) for data in batch])
        # 3. compute the gradients for the whole batch using back propagation
        nabla_b, nabla_w = self.back_propagation(input, target)
        # update biases and weights
        m = len(batch)
        self.biases = [
            b - (eta/m) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]
        self.weights = [
            w - (eta/m) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
    
    def back_propagation(self, input, target):
        ###############################
        # 1. forward pass
        # 1a. create variables to store the activations and z vectors for each layer
        # activation starts with the input layer activations, which is just the input data
        activation = input
        activation_list = [input]  # initialize with input layer activations
        # weighted input start at the first hidden layer, so we initialize an empty list
        z_list = []
        # 1b. loop over layers in forward direction, starting from the first hidden layer,
        # compute and store the activations and z vectors layer by layer
        # for the whole batch at once.
        for b, w in zip(self.biases, self.weights):
            layer_size = b.shape[0]
            # b is a vector of shape (layer_size, 1), we need to broadcast it (stack it horizontally)
            # to match the shape of the dot product w @ activation, which is (layer_size, batch_size)
            B = np.broadcast_to(b, (layer_size, input.shape[1]))
            z = np.dot(w, activation) + B
            activation = self.activation_func(z)
            z_list.append(z)
            activation_list.append(activation)
        ###############################
        # now we have all the information we need to compute the gradients in the backward pass.
        ###############################
        # 2. backward pass
        # 2a. create empty lists to store the gradients for biases and weights, layer by layer
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        # 2b. first compute the error "delta" for the output layer
        # this is what we called the "base case"
        delta = self.cost_derivative(activation_list[-1], target) * self.activation_prime(z_list[-1])
        # 2c. compute and store the gradients for the output layer
        # we use activation_list[-2] because the rule for updating the weights requires the activations from the previous layer,
        # which is the second to last layer in the list. The transpose is needed to match the dimensions of the matrices
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activation_list[-2].T)
        # 2c. loop over layers in reverse order,
        # compute the gradients for each layer. This is the "inductive step".
        # the loop starts at the second to last layer, and goes backwards to the first hidden layer.
        for l in range(2, self.number_of_layers+1):
            z = z_list[-l]
            # the order of the dot product and the transpose of the weights are needed
            # to match the dimensions of the matrices
            delta = np.dot(self.weights[-l+1].T, delta) * self.activation_prime(z)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activation_list[-l-1].T)
        return nabla_b, nabla_w
    
    def evaluate_old(self, test_data):
        """Return the number of correct classifications."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate(self, test_data):
        """Return the number of correct classifications.
        Works whether y is a scalar label or a one-hot encoded vector.
        """
        test_results = []
        for (x, y) in test_data:
            # Get the network's prediction
            prediction = np.argmax(self.feedforward(x))
            
            # Determine the true label
            # If y is iterable (one-hot), get the index of the 1.
            # Otherwise, assume y is already the integer label.
            true_label = np.argmax(y) if np.ndim(y) > 0 else y
            
            test_results.append((prediction, true_label))

        return sum(int(x == y) for (x, y) in test_results)

def load_mnist_from_web():
    urls = {
        "train_img": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_lbl": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_img": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_lbl": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    }

    data_results = {}

    for key, url in urls.items():
        print(f"Fetching {key}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Wrap the content in BytesIO and decompress in memory
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            if "img" in key:
                # Images: offset 16
                data_results[key] = np.frombuffer(f.read(), np.uint8, offset=16)
            else:
                # Labels: offset 8
                data_results[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    # Re-format to Nielsen's expected structure
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    training_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in data_results["train_img"].reshape(-1, 784)]
    training_results = [vectorized_result(y) for y in data_results["train_lbl"]]
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in data_results["test_img"].reshape(-1, 784)]
    test_data = list(zip(test_inputs, data_results["test_lbl"]))

    return training_data, test_data