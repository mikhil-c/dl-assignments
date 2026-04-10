"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils import data_loader

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="Choose between mnist and fashion_mnist"
    )
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "-l", "--loss",
        type=str,
        default="cross_entropy",
        choices=["mean_squared_error", "cross_entropy"],
        help="Choice of mean_squared_error or cross_entropy"
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str,
        default="rmsprop",
        choices=["sgd", "momentum", "nag", "rmsprop"],
        help="One of sgd, momentum, nag, rmsprop"
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight dacay for L2 regularization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=[128, 64, 32], nargs="+", help="Number of neurons in each hidden layer")
    parser.add_argument(
        "-a", "--activation",
        type=str,
        default="relu",
        choices=["sigmoid", "tanh", "relu"],
        help="Choice of sigmoid, tanh, relu for every hidden layer"
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"],
        help="Choice of random or xavier"
    )
    parser.add_argument(
        "-w_p", "--wandb_project",
        type=str,
        default="MLP-for-Image-Classification",
        help="Weights and Biases Project ID"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/best_model.npy",
        help="Relative path to the saved best model weights"
    )
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """

    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    return model.evaluate(X_test, y_test)


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    args = parse_arguments()

    (_, _), (X_test, y_test) = data_loader.get_data(args.dataset)
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    results = evaluate_model(model, X_test, y_test)

    print("-" * 40)
    print("Evaluation Complete")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 40)

    return results


if __name__ == '__main__':
    main()
