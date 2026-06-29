import joblib  # Для сохранения обученной модели
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Загрузка данных
digits = load_digits()
X = digits.data    # 64 пикселя (входы для нейронов)
y = digits.target  # Правильные ответы от 0 до 9

# === 1: Разделить данные на обучение и тест ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# test_size=0.2, на тест ушло 20%, а на обучение осталось 80%
# X_train - 80% картинок для обучения
# y_train - Обучающие ответы
# X_test: 20% картинок, которые мы прячем от модели во время обучения
# y_test: Правильные ответы к картинкам из X_test

# === 2: Создать модель (слой логистических нейронов) ===
model = LogisticRegression(max_iter=10000)

print("Обучение модели запущенно...")
# === 3: Обучить модель ===
model.fit(X_train, y_train)
print("Модель успешно обучена!")

# Сохраняем модель и тестовые данные для следующего скрипта
joblib.dump(model, "digits_model.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")
print("Модель сохранена в файл 'digits_model.pkl'")