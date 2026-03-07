"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
from ann.neural_network import NeuralNetwork
from utils import data_loader

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist"],
        help="Choose between mnist and fashion_mnist"
    )
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Mini-batch size")
    parser.add_argument(
        "-l", "--loss",
        type=str,
        required=True,
        choices=["mean_squared_error", "cross_entropy"],
        help="Choice of mean_squared_error or cross_entropy"
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str,
        required=True,
        choices=["sgd", "momentum", "nag", "rmsprop"],
        help="One of sgd, momentum, nag, rmsprop"
    )
    parser.add_argument("-lr", "--learning_rate", type=float, required=True, help="Initial learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight dacay for L2 regularization")
    parser.add_argument("-nhl", "--num_layers", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, required=True, nargs="+", help="Number of neurons in each hidden layer")
    parser.add_argument(
        "-a", "--activation",
        type=str,
        required=True,
        choices=["sigmoid", "tanh", "relu"],
        help="Choice of sigmoid, tanh, relu for every hidden layer"
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        required=True,
        choices=["random", "xavier"],
        help="Choice of random or xavier"
    )
    parser.add_argument(
        "-w_p", "--wandb_project",
        type=str,
        default="MLP-for-Image-Classification",
        help="Weights and Biases Project ID"
    )

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    (X_train, y_train), (_, _) = data_loader.get_data(args.dataset)

    mlp = NeuralNetwork(args)
    mlp.train(
        X_train=X_train, 
        y_train=y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )

    print("Training complete!")

    best_weights = mlp.get_weights()
    np.save("src/best_model.npy", best_weights)

    with open("src/best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    main()
