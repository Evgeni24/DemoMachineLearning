import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Загружаем датасет
digits = load_digits()

# 2. Находим индексы всех картинок, где целевая переменная (target) равна 2
twos_indices = np.where(digits.target == 5)[0]

# Берем первые 10 уникальных двоек
selected_twos = twos_indices[:10]

# 3. Строим сетку для отображения (1 строка, 10 колонок)
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 3))

for i, idx in enumerate(selected_twos):
    image = digits.images[idx]
    ax = axes[i]

    # Отображаем картинку
    ax.imshow(image, cmap="gray_r")

    # Подписываем индекс картинки в датасете для интерактивности
    ax.set_title(f"Вариант №{i + 1}\n(id: {idx})", fontsize=10)

    # Отключаем оси
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Разнообразие написания одной и той же цифры '2' в датасете", fontsize=16, y=1.1)
plt.tight_layout()
plt.show()