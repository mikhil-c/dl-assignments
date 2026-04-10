import sys
sys.path.insert(0, 'src')

import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ann.neural_network import NeuralNetwork
from utils.data_loader import get_data

(_, _), (X_test, y_test) = get_data("mnist")

# Load best model
def load_model(path):
    return np.load(path, allow_pickle=True).item()

class Args:
    dataset = "mnist"
    loss = "cross_entropy"
    wandb_project = "MLP-for-Image-Classification"
    batch_size = 32
    learning_rate = 0.001
    optimizer = "rmsprop"
    num_layers = 3
    hidden_size = [128, 64, 32]
    activation = "relu"
    weight_init = "xavier"
    weight_decay = 0.0

wandb.init(project="MLP-for-Image-Classification", name="2.8-confusion-matrix")

model = NeuralNetwork(Args())
weights = load_model("src/best_model.npy")
model.set_weights(weights)

results = model.evaluate(X_test, y_test)
logits = results["logits"]
y_pred = np.argmax(logits, axis=1)
y_true = np.argmax(y_test, axis=1)

# Standard confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, colorbar=True)
ax.set_title("Confusion Matrix — Best Model")
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close()

# Creative visualization — most confused pairs
print("Most confused pairs:")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    print(f"  True={idx[0]} predicted as {idx[1]}: {cm_no_diag[idx]} times")
    cm_no_diag[idx] = 0

# Creative: show actual misclassified images
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for true_class in range(10):
    wrong_mask = (y_true == true_class) & (y_pred != y_true)
    wrong_indices = np.where(wrong_mask)[0]
    if len(wrong_indices) > 0:
        idx = wrong_indices[0]
        axes[0, true_class].imshow(X_test[idx].reshape(28,28), cmap="gray")
        axes[0, true_class].set_title(f"True:{true_class}\nPred:{y_pred[idx]}", fontsize=8)
        axes[0, true_class].axis("off")
    
    # show correctly classified for comparison
    right_mask = (y_true == true_class) & (y_pred == y_true)
    right_indices = np.where(right_mask)[0]
    if len(right_indices) > 0:
        idx = right_indices[0]
        axes[1, true_class].imshow(X_test[idx].reshape(28,28), cmap="gray")
        axes[1, true_class].set_title(f"Correct:{true_class}", fontsize=8)
        axes[1, true_class].axis("off")

fig.suptitle("Top: Misclassified | Bottom: Correctly Classified", fontsize=12)
wandb.log({"misclassified_vs_correct": wandb.Image(fig)})
plt.close()

print(f"Accuracy: {results['accuracy']*100:.2f}%")
print(f"F1: {results['f1']:.4f}")
wandb.finish()
print("Done!")
