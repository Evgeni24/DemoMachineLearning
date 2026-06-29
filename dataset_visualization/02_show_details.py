import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Загружаем данные
digits = load_digits()

# Возьмем для примера самое первое изображение (это цифра 0)
number = 0
image = digits.images[number]
target = digits.target[number]

# 2. Создаем красивую наглядную визуализацию
plt.figure(figsize=(10, 5))

# Слева: показываем саму картинку, как её видит человек
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray_r")  # gray_r инвертирует цвета, чтобы цифра была темной на белом фоне
plt.title(f"Как видит человек (Цифра: {target})", fontsize=14)
plt.axis("off")  # Убираем оси координат, чтобы выглядело как картинка

# Справа: показываем матрицу чисел, как её видит компьютер
plt.subplot(1, 2, 2)
plt.imshow(image, cmap="Blues", alpha=0.3)  # Легкий фон для наглядности сетки
plt.title("Как видит компьютер (Матрица 8x8)", fontsize=14)

# Записываем значение яркости (от 0 до 16) прямо поверх каждого пикселя
for i in range(8):
    for j in range(8):
        plt.text(
            j,
            i,
            int(image[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

plt.xticks(range(8))
plt.yticks(range(8))
plt.grid(False)
plt.tight_layout()
plt.show()

# 3. Демонстрируем, во что картинка превращается для модели (вектор)
print("=" * 60)
print(f"А вот так эта матрица вытягивается в один вектор (длиной {len(digits.data[0])} элемента) для обучения:")
print(digits.data[0].astype(int))
print("=" * 60)