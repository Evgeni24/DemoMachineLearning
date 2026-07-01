import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. ЗАГРУЗКА ДАННЫХ
file_path = "dirty_apartment_rent_data.xlsx"
df = pd.read_excel(file_path)

print(f"Исходный датасет загружен. Формат: {df.shape}")

# ==========================================
# ОЧИСТКА ДАННЫХ (Пропуски, баги, выбросы)
# ==========================================
# Удаляем строки без цены
df = df.dropna(subset=["price_per_month"])

# Исправляем отрицательную площадь
df["area_sqm"] = df["area_sqm"].abs()

# Заменяем прочерки в метро на NaN и заполняем медианой
df["metro_min"] = df["metro_min"].replace("—", pd.NA)
metro_median = df["metro_min"].dropna().median()
df["metro_min"] = df["metro_min"].fillna(metro_median).astype(float)

# Удаляем экстремальные выбросы по цене массового рынка
df = df[df["price_per_month"] <= 500000]

# ==========================================
# КОДИРОВАНИЕ ТЕКСТА (One-Hot Encoding)
# ==========================================
df_encoded = pd.get_dummies(df, columns=["repair"], dtype=int)

# ==========================================
# МАСШТАБИРОВАНИЕ ПРИЗНАКОВ С СОХРАНЕНИЕМ ID
# ==========================================
# Выносим id в отдельную переменную, чтобы он не участвовал в масштабировании
apartment_ids = df_encoded["id"].reset_index(drop=True)

# Теперь разделяем фичи (X) и таргет (y), полностью исключая из них id
X = df_encoded.drop(columns=["id", "price_per_month"]).reset_index(drop=True)
y = df_encoded[["price_per_month"]].reset_index(drop=True)

# Масштабируем входные признаки (X) в диапазон 0..1
scaler_X = MinMaxScaler()
X_scaled_array = scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

# Стандартизируем цену (y)
scaler_y = StandardScaler()
y_scaled_array = scaler_y.fit_transform(y)
y_scaled = pd.DataFrame(y_scaled_array, columns=y.columns)

# Склеиваем всё обратно в одну таблицу: сохраняем оригинальный ID,
# а также добавляем отмасштабированные фичи и таргет
df_final = pd.concat([apartment_ids, X_scaled, y_scaled], axis=1)

# ==========================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ==========================================
output_filename = "clean_apartment_rent_data.xlsx"
df_final.to_excel(output_filename, index=False)
# TODO: сохраняем коэффициент масштабирования
# joblib.dump(scaler_y, "price_scaler.pkl")

print(f"\nИдеально чистый датасет успешно сохранен в '{output_filename}'!")
print("\nПервые 3 строки финальной таблицы (ID остался оригинальным):")
print(df_final.head(3))