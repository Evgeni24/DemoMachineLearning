import numpy as np
import pandas as pd

# Фиксируем сид для воспроизводимости результатов
np.random.seed(42)

num_rows = 10000

# 1. Генерируем базовые чистые данные
areas = np.random.randint(28, 120, size=num_rows)  # Площадь от 28 до 120 кв.м.


# Логика комнат в зависимости от площади
def get_rooms(area):
    if area < 45:
        return 1
    elif area < 75:
        return 2
    else:
        return np.random.choice([3, 4], p=[0.7, 0.3])


rooms = [get_rooms(a) for a in areas]

repairs = np.random.choice(
    ["Косметический", "Бабушкин", "Евроремонт", "Дизайнерский"],
    size=num_rows,
    p=[0.4, 0.2, 0.3, 0.1],
)

metro_distance = np.random.randint(2, 30, size=num_rows)  # минут до метро

# Базовая цена, зависящая от параметров (с шумом)
base_prices = (
    areas * 1000 + np.array(rooms) * 5000 + (35 - metro_distance) * 500
)
# Добавим случайный разброс цен
prices = base_prices + np.random.randint(-10000, 15000, size=num_rows)
# Округлим цены до тысяч для красоты
prices = np.round(prices / 1000) * 1000

# Создаем начальный DataFrame
df = pd.DataFrame({
    "id": range(1, num_rows + 1),
    "area_sqm": areas,
    "rooms": rooms,
    "repair": repairs,
    "metro_min": metro_distance.astype(object),  # object, чтобы засунуть прочерки
    "price_per_month": prices.astype(float),  # float, чтобы поддерживать NaN
})

# 2. ИСКАЖАЕМ ДАННЫЕ (Вносим ошибки для демонстрации)

# Аномалия 1: Пропуски в целевом признаке (Цена) — ~5% строк
price_nan_idx = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
df.loc[price_nan_idx, "price_per_month"] = np.nan

# Аномалия 2: Прочерки вместо чисел в расстоянии до метро — ~7% строк
metro_dash_idx = np.random.choice(
    num_rows, size=int(num_rows * 0.07), replace=False
)
df.loc[metro_dash_idx, "metro_min"] = "—"

# Аномалия 3: Отрицательная площадь (технический баг) — 15 конкретных строк
negative_area_idx = np.random.choice(num_rows, size=15, replace=False)
df.loc[negative_area_idx, "area_sqm"] = df.loc[
    negative_area_idx, "area_sqm"
] * (-1)

# Аномалия 4: Экстремальные выбросы в цене (лишние нули / элитное жилье) — 20 строк
outlier_price_idx = np.random.choice(num_rows, size=20, replace=False)
df.loc[outlier_price_idx, "price_per_month"] = (
    df.loc[outlier_price_idx, "price_per_month"] * 10
)

# 3. Сохраняем в Excel
output_filename = "dirty_apartment_rent_data.xlsx"
df.to_excel(output_filename, index=False)

print(f"Файл '{output_filename}' успешно сгенерирован!")
print(f"Всего строк: {len(df)}")
print(f"Пропусков в цене (NaN): {df['price_per_month'].isna().sum()}")
print(f"Прочерков в метро ('—'): {df['metro_min'].eq('—').sum()}")
print(f"Отрицательных площадей: {df['area_sqm'].lt(0).sum()}")
print(
    f"Потенциальных выбросов (цена > 1 млн руб): {df['price_per_month'].gt(1000000).sum()}"
)