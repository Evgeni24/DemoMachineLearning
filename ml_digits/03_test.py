import random
import joblib
import matplotlib.pyplot as plt

# Загружаем сохраненную модель и тестовые данные
try:
    model = joblib.load("digits_model.pkl")
    X_test, y_test = joblib.load("test_data.pkl")
except FileNotFoundError:
    print("Ошибка: Сначала запустите и обучите модель в 02_train.py!")
    exit()

# === 1: Посчитать общую точность модели ===
accuracy = model.score(X_test, y_test)
print(f"\n[РЕЗУЛЬТАТ] Общая точность модели на тест-выборке: {accuracy * 100:.2f}%")

# === 2: Интерактивная проверка на случайной картинке ===
random_idx = random.randint(0, len(X_test) - 1)
sample_image = X_test[random_idx]
true_label = y_test[random_idx]

# Модель делает предсказание (ей нужен двумерный массив, поэтому оборачиваем в [])
predicted_label = model.predict([sample_image])[0]

print(f"Случайный индекс теста: {random_idx}")
print(f"Правильный ответ: {true_label}")
print(f"Предсказание модели: {predicted_label}")

# Показываем картинку
plt.figure(figsize=(4, 4))
# Возвращаем вектору из 64 чисел форму матрицы 8x8 для отрисовки
plt.imshow(sample_image.reshape(8, 8), cmap="gray_r")
plt.title(f"Факт: {true_label} | ИИ: {predicted_label}")
plt.axis("off")
plt.show()