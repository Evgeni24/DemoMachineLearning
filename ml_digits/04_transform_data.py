import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Путь к вашей картинке
image_path = "digit3.png"

if not os.path.exists(image_path):
    print(f"Ошибка: Файл '{image_path}' не найден в папке проекта.")
    print("Пожалуйста, сохраните туда вашу нарисованную цифру.")
    exit()

# 1. Загружаем оригинал для отображения
img_original = Image.open(image_path)

# 2. Пошаговая трансформация под формат Scikit-Learn (load_digits)
# Шаг A: Переводим в черно-белый режим (L)
img_gray = img_original.convert("L")

# Шаг B: Сжимаем до 8x8 пикселей
img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)

# Шаг C: Переводим в массив NumPy
img_array = np.array(img_resized)

# Шаг D: Инвертируем цвета и приводим к шкале от 0 до 16
# (В PIL 255 — белый фон, в sklearn 0 — белый фон)
img_scaled = (255.0 - img_array) / 255.0 * 16.0
img_final = np.round(img_scaled).astype(int)

# 3. Визуализация результата
plt.figure(figsize=(10, 5))

# Левый график: Оригинальное изображение
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title(f"Оригинал\nРазмер: {img_original.size[0]}x{img_original.size[1]}px")
plt.axis("off")

# Правый график: Что видит модель после трансформации
plt.subplot(1, 2, 2)
plt.imshow(img_final, cmap="Blues", alpha=0.3)
plt.title("После трансформации\nРазмер: 8x8px (Шкала 0..16)")

# Накладываем сетку и значения яркости пикселей поверх квадратов
for i in range(8):
    for j in range(8):
        plt.text(
            j,
            i,
            str(img_final[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

plt.xticks(range(8))
plt.yticks(range(8))
plt.grid(False)

plt.suptitle("Отладка предобработки изображения для ML", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# Выводим плоский вектор в консоль
print("\n" + "=" * 50)
print("Плоский вектор (длина 64), который уйдет в model.predict():")
print("=" * 50)
print(img_final.flatten())
print("=" * 50)