import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

# Находим первые 10 уникальных двоек
twos_indices = np.where(digits.target == 2)[0][:10]

fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 3))
for i, idx in enumerate(twos_indices):
    ax = axes[i]
    ax.imshow(digits.images[idx], cmap="gray_r")
    ax.set_title(f"Вариант №{i+1}")
    ax.axis("off")

plt.suptitle("Как разные люди пишут цифру 2 (Размер: 8x8 пикселей)", fontsize=14)
plt.tight_layout()
plt.show()