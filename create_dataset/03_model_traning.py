import joblib  # Для сохранения обученной модели
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Загрузка очищенных и отмасштабированных данных
file_path = "clean_apartment_rent_data.xlsx"
df = pd.read_excel(file_path)

# Выделяем входы (X) и выходы (y) для модели
# Исключаем 'id' (это просто номер строки) и целевой признак 'price_per_month'
X = df.drop(columns=["id", "price_per_month"])
y = df["price_per_month"]  # Наш таргет (отмасштабированная цена)

print(f"Данные загружены. Количество фич для обучения: {X.shape[1]}")
print(f"Список признаков: {list(X.columns)}")

# === Шаг 1: Разделить данные на обучение и тест (80% на 20%) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Шаг 2: Создать модель (линейный нейрон для прогнозирования цен) ===
# Для предсказания стоимости мы используем Линейную регрессию вместо Логистической
model = LinearRegression()

print("\nОбучение модели запущено...")
# === Шаг 3: Обучить модель ===
# На этом этапе алгоритм подбирает оптимальные веса (w) для каждого признака
model.fit(X_train, y_train)
print("Модель успешно обучена!")

# === Шаг 4: Демонстрация «интеллекта» модели на уроке ===
print("\n=== Результаты подбора весов нейрона ===")
print(f"Базовое смещение (w0 / bias): {model.intercept_:.4f}")
for feature, weight in zip(X.columns, model.coef_):
    print(f"Вес для признака '{feature}': {weight:.4f}")

# Сохраняем обученную модель и тестовые данные для этапа валидации (Пункта 5)
joblib.dump(model, "apartment_model.pkl")
joblib.dump((X_test, y_test), "apartment_test_data.pkl")
print("\nМодель и тестовые данные успешно сохранены!")