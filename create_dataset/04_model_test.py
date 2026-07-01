import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Загружаем сохраненную модель и тестовые данные
try:
    model = joblib.load("apartment_model.pkl")
    X_test, y_test = joblib.load("apartment_test_data.pkl")
    # TODO: загружаем, предварительно сохраненный коэффициент масштабирования
    # scaler_y = joblib.load("price_scaler.pkl")
except FileNotFoundError:
    print(
        "Ошибка: Сначала запустите и обучите модель: 03_model_traning.py"
    )
    exit()

# === 1: Посчитать метрику качества (R^2 Score) ===
# В регрессии .score() возвращает коэффициент детерминации R^2 (от 0 до 1)
r2_score = model.score(X_test, y_test)
print(
    f"\n[РЕЗУЛЬТАТ] Коэффициент детерминации R^2 на тест-выборке: {r2_score:.4f}"
)
print(
    f"Это значит, что модель объясняет {r2_score * 100:.2f}% дисперсии цен на рынке."
)
print("-" * 50)

# === 2: Проверка на одной случайной квартире ===
random_idx = random.randint(0, len(X_test) - 1)

# Вытаскиваем строку по её порядковому номеру через .iloc
sample_features = X_test.iloc[random_idx]
true_price_scaled = y_test.iloc[random_idx]

# Модель делает предсказание (ожидает таблицу/двумерный массив, передаем как DataFrame одной строки)
sample_df = pd.DataFrame([sample_features])
predicted_price_scaled = model.predict(sample_df)[0]

# TODO: преобразуем цену назад в рубли
# real_price_rub = scaler_y.inverse_transform([[true_price_scaled]])[0][0]
# predicted_price_rub = scaler_y.inverse_transform([[predicted_price_scaled]])[0][0]

print(f"Случайный индекс в тестовой выборке: {random_idx}")
print(f"Фактическая цена (отмасштабированная): {true_price_scaled:.4f}")
# print(f"Фактическая цена (руб): {real_price_rub:.4f}")
print(f"Предсказание модели (отмасштабированное): {predicted_price_scaled:.4f}")
# print(f"Предсказание модели (руб): {predicted_price_rub:.4f}")
print("-" * 50)


# === 3: Визуализация «План vs Факт» для урока ===
print("Строим график сравнения реальных и предсказанных цен...")

# Делаем предсказание для ВСЕЙ тестовой выборки
all_predictions_scaled = model.predict(X_test)

# Для графика выберем первые 50 квартир из теста, чтобы график не превратился в кашу
num_samples_to_plot = 50

plt.figure(figsize=(14, 6))
plt.plot(
    range(num_samples_to_plot),
    y_test[:num_samples_to_plot],
    label="Реальная цена (Факт)",
    color="royalblue",
    marker="o",
    linewidth=2,
)
plt.plot(
    range(num_samples_to_plot),
    all_predictions_scaled[:num_samples_to_plot],
    label="Предсказание ИИ (Прогноз)",
    color="darkorange",
    linestyle="--",
    marker="x",
    linewidth=1.5,
)

plt.title(
    f"Сравнение реальных цен и предсказаний модели (Первые {num_samples_to_plot} квартир)"
)
plt.xlabel("Индекс квартиры в тесте")
plt.ylabel("Отмасштабированная цена")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()