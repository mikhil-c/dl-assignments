import sys
sys.path.insert(0, 'src')

import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import get_data
from sklearn.model_selection import train_test_split

# Load data once
(X_train_full, y_train_full), (X_test, y_test) = get_data("mnist")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Sweep config
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs":        {"values": [10, 20]},
        "batch_size":    {"values": [16, 32, 64]},
        "learning_rate": {"values": [0.001, 0.0001, 0.01]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "num_layers":    {"values": [1, 2, 3]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["relu", "tanh", "sigmoid"]},
        "weight_init":   {"values": ["xavier", "random"]},
        "weight_decay":  {"values": [0.0, 0.0005, 0.001]},
    }
}

def train():
    wandb.init()
    config = wandb.config

    # Build a simple args object
    class Args:
        dataset = "mnist"
        loss = "cross_entropy"
        wandb_project = "MLP-for-Image-Classification"
        epochs = config.epochs
        batch_size = config.batch_size
        learning_rate = config.learning_rate
        optimizer = config.optimizer
        num_layers = config.num_layers
        hidden_size = [config.hidden_size] * config.num_layers
        activation = config.activation
        weight_init = config.weight_init
        weight_decay = config.weight_decay

    model = NeuralNetwork(Args())

    for epoch in range(config.epochs):
        model.train(X_train, y_train, epochs=1, batch_size=config.batch_size)
        val_results = model.evaluate(X_val, y_val)
        train_results = model.evaluate(X_train, y_train)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_f1": val_results["f1"],
        })

    wandb.finish()

# Create and run sweep
sweep_id = wandb.sweep(sweep_config, project="MLP-for-Image-Classification")
wandb.agent(sweep_id, function=train, count=100)
