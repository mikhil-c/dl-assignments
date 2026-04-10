import sys
sys.path.insert(0, 'src')

import wandb
import numpy as np
from utils.data_loader import get_data

wandb.init(project="MLP-for-Image-Classification", name="2.1-data-exploration")

(X_train, y_train), _ = get_data("mnist")
y_indices = np.argmax(y_train, axis=1)

table = wandb.Table(columns=["image", "class"])

for class_idx in range(10):
    class_mask = np.where(y_indices == class_idx)[0]
    samples = np.random.choice(class_mask, size=5, replace=False)
    for sample_idx in samples:
        img = X_train[sample_idx].reshape(28, 28)
        table.add_data(wandb.Image(img), class_idx)

wandb.log({"sample_images": table})
wandb.finish()
print("Done!")
