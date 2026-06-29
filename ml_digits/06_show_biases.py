import joblib
import matplotlib.pyplot as plt

# Загружаем сохраненную модель и тестовые данные
try:
    model = joblib.load("digits_model.pkl")
    X_test, y_test = joblib.load("test_data.pkl")
except FileNotFoundError:
    print("Ошибка: Сначала запустите и обучите модель в 02_train.py!")
    exit()

# Получаем сдвиги (intercepts)
biases = model.intercept_

# Выводим их для каждого из 10 нейронов
for digit, bias in enumerate(biases):
    print(f"Нейрон цифры {digit}: Bias = {bias:.4f}")
    print(f"Cумму весов нейрона {model.coef_.sum(axis=1)}")