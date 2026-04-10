import sys
sys.path.insert(0, 'src')

import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import get_data
from sklearn.model_selection import train_test_split

(X_train_full, y_train_full), (X_test, y_test) = get_data("mnist")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

def run_experiment(name, args_dict):
    class Args:
        dataset = "mnist"
        loss = args_dict.get("loss", "cross_entropy")
        wandb_project = "MLP-for-Image-Classification"
        batch_size = args_dict.get("batch_size", 32)
        learning_rate = args_dict.get("learning_rate", 0.001)
        optimizer = args_dict.get("optimizer", "rmsprop")
        num_layers = args_dict.get("num_layers", 3)
        hidden_size = args_dict.get("hidden_size", [128, 128, 128])
        activation = args_dict.get("activation", "relu")
        weight_init = args_dict.get("weight_init", "xavier")
        weight_decay = args_dict.get("weight_decay", 0.0)

    num_epochs = args_dict.get("epochs", 10)

    wandb.init(
        project="MLP-for-Image-Classification",
        name=name,
        config=args_dict
    )

    model = NeuralNetwork(Args())

    for epoch in range(num_epochs):
        model.train(X_train, y_train, epochs=1, batch_size=Args.batch_size)
        val_results = model.evaluate(X_val, y_val)
        train_results = model.evaluate(X_train, y_train)

        # 2.4: gradient norm of first hidden layer
        grad_norm = np.linalg.norm(model._NeuralNetwork__layers[0].grad_W)

        # 2.5: fraction of dead neurons in each hidden layer
        dead_neurons = []
        for layer in model._NeuralNetwork__layers[:-1]:
            if layer.a is not None:
                dead = np.mean(layer.a == 0)
                dead_neurons.append(dead)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_f1": val_results["f1"],
            "grad_norm_layer0": grad_norm,
        }
        for i, d in enumerate(dead_neurons):
            log_dict[f"dead_neurons_layer{i}"] = d

        wandb.log(log_dict)

    wandb.finish()


# ── 2.3 Optimizer Showdown ──
print("Running 2.3 optimizer comparison...")
for opt in ["sgd", "momentum", "nag", "rmsprop"]:
    run_experiment(
        name=f"2.3-optimizer-{opt}",
        args_dict={
            "optimizer": opt,
            "activation": "relu",
            "num_layers": 3,
            "hidden_size": [128, 128, 128],
            "learning_rate": 0.001,
            "epochs": 10,
        }
    )

# ── 2.4 Vanishing Gradient ──
print("Running 2.4 vanishing gradient analysis...")
for activation in ["sigmoid", "relu"]:
    for hidden_size, num_layers in [([128, 128, 128], 3), ([64, 64, 64, 64], 4)]:
        run_experiment(
            name=f"2.4-activation-{activation}-layers-{num_layers}",
            args_dict={
                "optimizer": "rmsprop",
                "activation": activation,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "learning_rate": 0.001,
                "epochs": 10,
            }
        )

# ── 2.5 Dead Neuron Investigation ──
print("Running 2.5 dead neuron investigation...")
for activation in ["relu", "tanh"]:
    run_experiment(
        name=f"2.5-dead-neuron-{activation}-highlr",
        args_dict={
            "optimizer": "rmsprop",
            "activation": activation,
            "num_layers": 3,
            "hidden_size": [128, 128, 128],
            "learning_rate": 0.1,
            "epochs": 10,
        }
    )

# ── 2.6 Loss Function Comparison ──
print("Running 2.6 loss function comparison...")
for loss in ["cross_entropy", "mean_squared_error"]:
    run_experiment(
        name=f"2.6-loss-{loss}",
        args_dict={
            "optimizer": "rmsprop",
            "activation": "relu",
            "num_layers": 3,
            "hidden_size": [128, 128, 128],
            "learning_rate": 0.001,
            "loss": loss,
            "epochs": 10,
        }
    )

print("All experiments done!")
