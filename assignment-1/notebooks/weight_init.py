import sys
sys.path.insert(0, 'src')

import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import get_data

(X_train_full, y_train_full), _ = get_data("mnist")
X_train = X_train_full[:10000]
y_train = y_train_full[:10000]

def run_init_experiment(init_type):
    class Args:
        dataset = "mnist"
        loss = "cross_entropy"
        wandb_project = "MLP-for-Image-Classification"
        batch_size = 32
        learning_rate = 0.001
        optimizer = "rmsprop"
        num_layers = 3
        hidden_size = [128, 128, 128]
        activation = "relu"
        weight_init = init_type
        weight_decay = 0.0

    wandb.init(
        project="MLP-for-Image-Classification",
        name=f"2.9-init-{init_type}"
    )

    model = NeuralNetwork(Args())

    iteration = 0
    for i in range(0, 50 * 32, 32):
        X_batch = X_train[i:i+32]
        y_batch = y_train[i:i+32]

        logits = model.forward(X_batch)
        model.backward(y_batch, logits)
        model.update_weights(32)

        # grad_W shape (784, 128) — log gradient norm of 5 neurons
        gW = model._NeuralNetwork__layers[0].grad_W
        log_dict = {"iteration": iteration}
        for j in range(5):
            log_dict[f"neuron_{j}_grad_norm"] = float(np.linalg.norm(gW[:, j]))

        wandb.log(log_dict)
        iteration += 1

    wandb.finish()
    print(f"Done: {init_type}")

run_init_experiment("xavier")
run_init_experiment("zeros")
print("2.9 done!")
