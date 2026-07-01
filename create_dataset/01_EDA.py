# Exploratory Data Analysis(разведочный анализ данных)

import pandas as pd

# 1. Загрузка «грязных» данных
file_path = "dirty_apartment_rent_data.xlsx"
df = pd.read_excel(file_path)

print("=== ШАГ 1. ПЕРВЫЙ ВЗГЛЯД НА ДАННЫЕ ===")
print(df.head(10))
print("\n" + "=" * 40 + "\n")

print("=== ШАГ 2. ПРОВЕРКА ТИПОВ ДАННЫХ И ПРОПУСКОВ (df.info()) ===")
print(df.info())
print("\n" + "=" * 40 + "\n")

print("=== ШАГ 3. ПОИСК СТРАННОСТЕЙ В СТАТИСТИКЕ (df.describe()) ===")
print(df.describe())
print("\n" + "=" * 40 + "\n")


print("=== ШАГ 4. ПОД СЧЕТ КОНКРЕТНЫХ АНОМАЛИЙ ===")
# Считаем явные пропуски в цене
missing_prices = df["price_per_month"].isna().sum()
print(f"Пропущенных значений в колонке Цена: {missing_prices}")

# Считаем текстовые прочерки в колонке метро
dash_metro = (df["metro_min"] == "—").sum()
print(f"Текстовых прочерков '—' в колонке Метро: {dash_metro}")

# Считаем отрицательные площади
neg_area = (df["area_sqm"] < 0).sum()
print(f"Квартир с отрицательной площадью: {neg_area}")

# Считаем заоблачные цены
crazy_price = (df["price_per_month"] > 500000).sum()
print(f"Квартир с ценой аренды > 500 000 руб/мес: {crazy_price}")
print("\n" + "=" * 40 + "\n")

