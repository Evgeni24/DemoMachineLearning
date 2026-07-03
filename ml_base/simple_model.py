import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ==========================================
# ЧАСТЬ 1. АРХИТЕКТУРА СЕТИ НА ЧИСТОМ PYTHON
# ==========================================
class TrainableNeuron:

    def __init__(self, num_inputs: int):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        self.last_inputs: list[float] = []
        self.last_output: float = 0.0

    def forward(self, inputs: list[float]) -> float:
        self.last_inputs = list(inputs)
        weighted_sum = (
            sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        )
        weighted_sum = max(-500.0, min(500.0, weighted_sum))
        self.last_output = 1.0 / (1.0 + math.exp(-weighted_sum))
        return self.last_output

    def backward(self, error: float, learning_rate: float):
        sigmoid_derivative = self.last_output * (1.0 - self.last_output)
        delta = error * sigmoid_derivative
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * self.last_inputs[i]
        self.bias -= learning_rate * delta


class DigitClassifierModel:

    def __init__(self, num_inputs: int = 64, num_outputs: int = 10):
        self.neurons = [TrainableNeuron(num_inputs) for _ in range(num_outputs)]

    def predict(self, inputs: list[float]) -> list[float]:
        return [neuron.forward(inputs) for neuron in self.neurons]

    def train_on_sample(self, inputs: list[float], target_class: int, lr: float):
        predictions = self.predict(inputs)
        for digit_index, neuron in enumerate(self.neurons):
            target_output = 1.0 if digit_index == target_class else 0.0
            error = predictions[digit_index] - target_output
            neuron.backward(error, lr)


# ==========================================
# ЧАСТЬ 2. ПОДГОТОВКА ДАННЫХ И ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
# 1. Загрузка данных
digits = load_digits()

# Нормализуем пиксели [0..16] -> [0..1]
X = digits.data / 16.0
y = digits.target

# Разделяем на Train/Test (сохраняем структуру массивов numpy для matplotlib)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаем модель
model = DigitClassifierModel(num_inputs=64, num_outputs=10)

EPOCHS = 15
LEARNING_RATE = 0.1

print("=== 1. ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ЦИФР ===")
for epoch in range(EPOCHS):
    for inputs, target in zip(X_train, y_train):
        # В метод обучения передаем как обычный python-список
        model.train_on_sample(inputs.tolist(), target, lr=LEARNING_RATE)

    # Быстрая проверка точности на каждой эпохе
    correct = 0
    for inputs, target in zip(X_test, y_test):
        outputs = model.predict(inputs.tolist())
        if outputs.index(max(outputs)) == target:
            correct += 1
    accuracy = correct / len(X_test)
    print(f"Эпоха {epoch + 1:2d}/{EPOCHS} | Точность на тесте: {accuracy * 100:.2f}%")

print("\nМодель успешно обучена!")
print("-" * 50)


# ==========================================
# ЧАСТЬ 3. ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ И ПРОВЕРКА
# ==========================================
print("=== 2. СЛУЧАЙНАЯ ПРОВЕРКА И ОТРИСОВКА ГРАФИКА ===")

# Выбираем случайный индекс из тестовой выборки
random_idx = random.randint(0, len(X_test) - 1)
sample_image = X_test[random_idx]  # Массив из 64 чисел (пиксели)
true_label = y_test[random_idx]  # Правильный ответ (цифра)

# Наш ИИ делает предсказание
# Передаем массив в виде списка, находим индекс самого уверенного нейрона
neuron_outputs = model.predict(sample_image.tolist())
predicted_label = neuron_outputs.index(max(neuron_outputs))

print(f"Выбран случайный индекс теста: {random_idx}")
print(f"Правильный ответ (Факт): {true_label}")
print(f"Предсказание модели (ИИ): {predicted_label}")

# Отрисовка картинки 8x8 средствами matplotlib
plt.figure(figsize=(4, 4))
# Возвращаем вектору из 64 чисел форму матрицы 8x8 для отрисовки
plt.imshow(sample_image.reshape(8, 8), cmap="gray_r")
plt.title(f"Факт: {true_label} | ИИ: {predicted_label}", fontsize=14, color="green" if true_label == predicted_label else "red")
plt.axis("off")
plt.show()