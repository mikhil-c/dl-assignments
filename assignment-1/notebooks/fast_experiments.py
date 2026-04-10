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

# Use small subset for speed
X_tr = X_train[:10000]
y_tr = y_train[:10000]

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

    num_epochs = args_dict.get("epochs", 5)

    wandb.init(project="MLP-for-Image-Classification", name=name, config=args_dict)
    model = NeuralNetwork(Args())

    for epoch in range(num_epochs):
        model.train(X_tr, y_tr, epochs=1, batch_size=Args.batch_size)
        val_results = model.evaluate(X_val, y_val)
        train_results = model.evaluate(X_tr[:2000], y_tr[:2000])

        grad_norm = np.linalg.norm(model._NeuralNetwork__layers[0].grad_W)

        dead_neurons = []
        for layer in model._NeuralNetwork__layers[:-1]:
            if layer.a is not None:
                dead_neurons.append(float(np.mean(layer.a == 0)))

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
        print(f"  {name} epoch {epoch+1}/{num_epochs} val_acc={val_results['accuracy']:.3f}")

    wandb.finish()

# 2.3 Optimizer Showdown
print("=== 2.3 Optimizer Showdown ===")
for opt in ["sgd", "momentum", "nag", "rmsprop"]:
    run_experiment(f"2.3-optimizer-{opt}", {
        "optimizer": opt, "activation": "relu",
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "learning_rate": 0.001, "epochs": 5,
    })

# 2.4 Vanishing Gradient
print("=== 2.4 Vanishing Gradient ===")
for act in ["sigmoid", "relu"]:
    run_experiment(f"2.4-{act}", {
        "optimizer": "rmsprop", "activation": act,
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "learning_rate": 0.001, "epochs": 5,
    })

# 2.5 Dead Neuron
print("=== 2.5 Dead Neuron ===")
for act in ["relu", "tanh"]:
    run_experiment(f"2.5-{act}-highlr", {
        "optimizer": "rmsprop", "activation": act,
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "learning_rate": 0.1, "epochs": 5,
    })

# 2.6 Loss Comparison
print("=== 2.6 Loss Comparison ===")
for loss in ["cross_entropy", "mean_squared_error"]:
    run_experiment(f"2.6-{loss}", {
        "optimizer": "rmsprop", "activation": "relu",
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "learning_rate": 0.001, "loss": loss, "epochs": 5,
    })

print("=== 2.9 Weight Init ===")
for init in ["xavier", "random"]:
    run_experiment(f"2.9-init-{init}", {
        "optimizer": "rmsprop", "activation": "relu",
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "learning_rate": 0.001, "weight_init": init, "epochs": 5,
    })

print("=== 2.10 Fashion MNIST ===")
(X_fashion_full, y_fashion_full), (X_fashion_test, y_fashion_test) = get_data("fashion_mnist")
X_fashion_train = X_fashion_full[:10000]
y_fashion_train = y_fashion_full[:10000]

for config in [
    {"optimizer": "rmsprop", "activation": "relu",    "hidden_size": [128,128,128], "num_layers": 3},
    {"optimizer": "rmsprop", "activation": "tanh",    "hidden_size": [128,128,128], "num_layers": 3},
    {"optimizer": "momentum","activation": "relu",    "hidden_size": [128,64,32],   "num_layers": 3},
]:
    name = f"2.10-fashion-{config['optimizer']}-{config['activation']}"
    wandb.init(project="MLP-for-Image-Classification", name=name, config=config)
    class Args:
        dataset = "fashion_mnist"
        loss = "cross_entropy"
        wandb_project = "MLP-for-Image-Classification"
        batch_size = 32
        learning_rate = 0.001
        optimizer = config["optimizer"]
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        activation = config["activation"]
        weight_init = "xavier"
        weight_decay = 0.0
    model = NeuralNetwork(Args())
    for epoch in range(5):
        model.train(X_fashion_train, y_fashion_train, epochs=1, batch_size=32)
        val_res = model.evaluate(X_fashion_test, y_fashion_test)
        wandb.log({"epoch": epoch+1, "val_accuracy": val_res["accuracy"], "val_f1": val_res["f1"]})
        print(f"  {name} epoch {epoch+1}/5 val_acc={val_res['accuracy']:.3f}")
    wandb.finish()

print("ALL DONE!")
