"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import Layer
from . import activations
from . import objective_functions
from . import optimizers

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """

        self.__layers = []
        if cli_args.num_layers == 0:
            self.__layers.append(Layer(784, 10, cli_args.weight_init))
        else:
            self.__layers.append(Layer(784, cli_args.hidden_size[0], cli_args.weight_init))
            for i in range(1, cli_args.num_layers):
                self.__layers.append(Layer(cli_args.hidden_size[i - 1], cli_args.hidden_size[i], cli_args.weight_init))
            self.__layers.append(Layer(cli_args.hidden_size[-1], 10, cli_args.weight_init))

        self.__activation = cli_args.activation
        self.__loss = cli_args.loss
        self.__weight_decay = cli_args.weight_decay

        __opt_map = {
            "sgd": optimizers.SGD(cli_args.learning_rate),
            "momentum": optimizers.Momentum(cli_args.learning_rate),
            "nag": optimizers.NAG(cli_args.learning_rate),
            "rmsprop": optimizers.RMSProp(cli_args.learning_rate)
        }
        self.__optimizer = __opt_map[cli_args.optimizer]

    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        layer_output = X
        for i in range(0, len(self.__layers) - 1):
            layer_output = self.__layers[i].forward(layer_output, self.__activation)

        self.__output_logits = self.__layers[-1].forward(layer_output, "identity")
        return self.__output_logits
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs  # output logits for mse
                                       # activation values obtained after applying softmax for cross_entropy
            
        Returns:
            return grad_w, grad_b
        """

        if self.__loss == "cross_entropy":
            local_grad = y_pred - y_true
        else:
            local_grad = objective_functions.backward(y_pred, y_true, self.__loss) 

        _ = self.__layers[-1].backward(local_grad, self.__weight_decay) # Initializes the local gradient at the output layer
        w_next = self.__layers[-1].W
        grad_W, grad_b = [], []
        grad_W.append(self.__layers[-1].grad_W.T)
        grad_b.append(self.__layers[-1].grad_b.T)
        for layer in self.__layers[-2::-1]:
            local_grad = layer.backward(local_grad, self.__weight_decay, self.__activation, w_next)
            grad_W.append(layer.grad_W.T)
            grad_b.append(layer.grad_b.T)
            w_next = layer.W

        return (grad_W, grad_b)
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """

        for i, layer in enumerate(self.__layers):
            layer.W = self.__optimizer.update(layer.W, layer.grad_W, f"W{i}")
            layer.b = self.__optimizer.update(layer.b, layer.grad_b, f"b{i}")

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Train the network for specified epochs.
        """

        N = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X = X_train[indices]
            y = y_train[indices]

            for i in range(0, N, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                logits = self.forward(X_batch)
                if self.__loss == "cross_entropy":
                    y_pred = activations.forward("softmax", logits)
                else:
                    y_pred = logits
                self.backward(y_batch, y_pred)
                self.update_weights()

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """

        logits = self.forward(X)
        if self.__loss == "cross_entropy":
            y_pred = activations.forward("softmax", logits)
        else:
            y_pred = logits

        l2_norm = sum(np.sum(layer.W ** 2) for layer in self.__layers)
        loss_val = objective_functions.forward(y_pred, y, self.__loss, self.__weight_decay, l2_norm)

        y_true_idx = np.argmax(y, axis=1)
        y_pred_idx = np.argmax(y_pred, axis=1)
        
        accuracy = np.mean(y_true_idx == y_pred_idx)

        num_classes = y.shape[1]
        precisions = []
        recalls = []
        
        for c in range(num_classes):
            tp = np.sum((y_pred_idx == c) & (y_true_idx == c))
            fp = np.sum((y_pred_idx == c) & (y_true_idx != c))
            fn = np.sum((y_pred_idx != c) & (y_true_idx == c))
            
            precisions.append(tp / (tp + fp + 1e-12))
            recalls.append(tp / (tp + fn + 1e-12))
            
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        return {
            "logits": logits,
            "loss": loss_val,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.__layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.__layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
