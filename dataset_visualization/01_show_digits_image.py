import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Загружаем датасет
digits = load_digits()

# 2. Настраиваем сетку: 1 строка, 10 колонок
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 4))

# В датасете первые 10 элементов как раз идут по порядку от 0 до 9
for i in range(10):
    image = digits.images[i]
    target = digits.target[i]
    ax = axes[i]

    # Отображаем саму картинку (инвертированная черно-белая палитра)
    ax.imshow(image, cmap="gray_r")
    ax.set_title(f"Цифра: {target}", fontsize=12, pad=10)

    # Убираем стандартные оси, чтобы они не мешали
    ax.set_xticks([])
    ax.set_yticks([])

    # Для первой цифры (0) оставим сетку пикселей, чтобы напомнить масштаб 8x8
    if i == 0:
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="gray", alpha=0.5, linestyle="--")

plt.suptitle("Примеры изображений из датасета (Размер каждого: 8x8 пикселей)", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# 3. Короткий интерактивный интерактив для студентов в консоли
print("=" * 70)
print("Размерность всего массива картинок (X):", digits.images.shape)
print("Это означает: 1797 картинок, каждая размером 8 на 8 пикселей.")
print("=" * 70)