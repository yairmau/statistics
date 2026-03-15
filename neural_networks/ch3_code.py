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


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.mean(
            np.nan_to_num(    # ensures numerical stability
                - y * np.log(a) - (1-y) * np.log(1-a)
            )
        )
    
    @staticmethod
    def delta(z, a, y, activation):   # z is kept for compatibility / consistency
        if activation == SigmoidActivation:
            return (a - y)
        return (a - y) / (a * (1 - a)) * activation.prime(z)

class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return np.linalg.norm(a-y)**2 / 2
    
    @staticmethod
    def delta(z, a, y, activation):
        return (a - y) * activation.prime(z)

class SigmoidActivation:
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        s = SigmoidActivation.fn(z)
        return s * (1 - s)

class ReLUActivation:
    @staticmethod
    def fn(z):
        return np.maximum(0, z)
    
    @staticmethod
    def prime(z):
        return np.where(z > 0, 1.0, 0.0)

class NN2:
    def __init__(self, layer_sizes, rand_seed=0, activation=SigmoidActivation, cost=CrossEntropyCost):
        """Initialize the neural network with the given layer sizes.
        For example, if layer_sizes = [50, 15, 20, 10], then we have a
        network with 50 input neurons, 15 neurons in hidden layer 0,
        20 neurons in hidden layer 1, and 10 output neurons (layer 3).
        """
        self.number_of_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(seed=rand_seed)
        self.activation = activation
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        # input layer has no weights nor biases, so we skip it.

        # each neuron get 1 bias, so bias vector has the size of the layer
        # we skip the first (input) layer, it doesn't have biases
        # I made the biases to be matrices of shape (N_b, 1) instead of vectors of shape (N_b,)
        # to make the broadcasting work more smoothly in the feedforward and backpropagation functions.
        self.biases = [self.rng.normal(loc=0, scale=1, size=(N_b, 1))
                       for N_b in self.layer_sizes[1:]]
        # each neuron in layer Right is connected to all neurons in layer Left,
        # so weight matrix has the shape (size_right, size_left)
        # again, we skip the first (input) layer, it doesn't have weights
        # scale_for_weights = 1 / np.sqrt(size_left)
        self.weights = [self.rng.normal(loc=0, scale=1/np.sqrt(size_left), size=(size_right, size_left))
                        for size_left, size_right in
                          zip(self.layer_sizes[:-1],self.layer_sizes[1:])
                        ]
    
    def large_weight_initializer(self):
        """same logic as above"""
        self.biases = [self.rng.normal(loc=0, scale=1, size=(N_b, 1))
                       for N_b in self.layer_sizes[1:]]
        self.weights = [self.rng.normal(loc=0, scale=1, size=(size_right, size_left))
                        for size_left, size_right in
                          zip(self.layer_sizes[:-1],self.layer_sizes[1:])
                        ]

    
    def feedforward(self, a):
        """given input `a` from the first layer,
           we sequencially compute the activations of each layer
           `feedforward` returns the activations of last (output) layer
        """
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.activation.fn(z)
        return a
    
    def stochastic_gradient_descent(self,
                                    training_data,
                                    test_data,
                                    epochs,
                                    batch_size,
                                    eta,
                                    lmbda_L1 = 0.0,
                                    lmbda_L2 = 0.0,
                                    dropout_prob=0.0,
                                    monitoring=False):
        n = len(training_data)
        lambda_L1_over_n = lmbda_L1 / n
        lambda_L2_over_n = lmbda_L2 / n
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
                    self.update_params_batch(batch, eta, lambda_L1_over_n, lambda_L2_over_n, dropout_prob)
                duration_seconds = time.perf_counter() - start_time
                test_acc = self.evaluate_vectorized(x_test, y_test) / len(test_data)
                pbar.set_description(f"Epoch: {epoch_j}/{epochs}")
                pbar.set_postfix_str(
                    f"test acc = {test_acc:.2%}, time/epoch = {duration_seconds:.2f}s"
                )
                if monitoring:
                    train_acc = self.evaluate_vectorized(x_train, y_train) / n
                    monitoring_dict["epoch"].append(epoch_j)
                    monitoring_dict["runtime_seconds"].append(duration_seconds)
                    monitoring_dict["train_accuracy"].append(train_acc)
                    monitoring_dict["test_accuracy"].append(test_acc)
                    monitoring_dict["train_cost"].append(self.cost.fn(self.feedforward(x_train), y_train))
                    monitoring_dict["test_cost"].append(self.cost.fn(self.feedforward(x_test), y_test))
            if monitoring:
                return monitoring_dict
        except KeyboardInterrupt:
            if 'pbar' in locals():
                pbar.close()
            print("\n\nTraining interrupted by user. Weights preserved.")
    
    def update_params_batch(self, batch, eta, lambda_L1_over_n, lambda_L2_over_n, dropout_prob):
        # 1. make input matrix, use column-major order, so each column is a training example, and each row is a feature.
        # input shape: (784, batch_size)
        input = np.hstack([data[0].reshape(-1, 1) for data in batch])
        # 2. make label matrix, use column-major order, so each column is a training example, and each row is a label.
        # target shape: (10, batch_size)
        target = np.hstack([data[1].reshape(-1, 1) for data in batch])
        # 3. create dropout masks for the hidden layers, shape: (layer_size, batch_size)
        dropout_activation_masks = []
        for layer in self.layer_sizes[1:-1]:
            mask = self.rng.random(size=(layer, len(batch)))
            mask = (mask > dropout_prob) / (1 - dropout_prob)
            dropout_activation_masks.append(mask)
        # 4. compute the gradients for the whole batch using back propagation
        nabla_b, nabla_w = self.back_propagation(input, target, dropout_activation_masks)
        # update biases and weights
        m = len(batch)
        self.biases = [
            b - (eta/m) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]
        self.weights = [
            w * (1 - eta * lambda_L2_over_n) - eta * lambda_L1_over_n * np.sign(w) - (eta/m) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
    
    def back_propagation(self, input, target, dropout_activation_masks):
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
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # numpy broadcasts b on its own
            z = np.dot(w, activation) + b
            activation = self.activation.fn(z)
            if i < self.number_of_layers - 1:
                # apply dropout mask to the activations of the hidden layers
                activation *= dropout_activation_masks[i]
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
        delta = self.cost.delta(z_list[-1], activation_list[-1], target, self.activation)
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
            # weights needed are the ones to the RIGHT of the current layer
            w_right = self.weights[-l+1]
            # activations needed are the ones to the LEFT of the current layer
            a_left = activation_list[-l-1]
            # the order of the dot product and the transpose of the weights are needed
            # to match the dimensions of the matrices
            delta = np.dot(w_right.T, delta) * self.activation.prime(z)
            # apply dropout mask to the delta values of the hidden layers
            if l <= self.number_of_layers:
                delta *= dropout_activation_masks[-l+1]
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, a_left.T)
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

    def evaluate_vectorized(self, x_matrix, y_matrix):
        """
        x_matrix: shape (784, N)
        y_matrix: shape (10, N)
        """
        # 1. get all predictions at once, shape: (N,)
        predictions = np.argmax(self.feedforward(x_matrix), axis=0)
        # 2. get true labels, shape: (N,)
        # y is one-hot (10, N), get the index of the 1
        true_labels = np.argmax(y_matrix, axis=0)
        # 3. sum the matches
        return np.sum(predictions == true_labels)


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
    test_results = [vectorized_result(y) for y in data_results["test_lbl"]]
    test_data = list(zip(test_inputs, test_results))

    # the label of both training and test are already one-hot encoded
    return training_data, test_data