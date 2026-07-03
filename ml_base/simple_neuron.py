class Neuron:

    def __init__(self, weights: list[float], bias: float):
        """Инициализация нейрона.

        :param weights: Список весов (w), по одному на каждый входной признак
        :param bias: Смещение (b), базовый уровень цены
        """
        self.weights = weights
        self.bias = bias

    def forward(self, inputs: list[float]) -> float:
        """Прямой проход (расчет предсказания нейрона).

        Формула: y = (x1 * w1) + (x2 * w2) + ... + b
        """
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Количество входных признаков должно совпадать с количеством весов!"
            )

        # Вычисляем взвешенную сумму: перемножаем входы на веса и складываем
        weighted_sum = 0.0
        for x, w in zip(inputs, self.weights):
            weighted_sum += x * w

        # Прибавляем смещение (bias)
        prediction = weighted_sum + self.bias
        return prediction


# ==========================================
# ДЕМОНСТРАЦИЯ РАБОТЫ НА УРОКЕ
# ==========================================
if __name__ == "__main__":
    # Представим, что модель уже обучилась и подобрала следующие коэффициенты:
    # Вес 1 (Площадь): +1.5 (чем больше площадь, тем выше цена)
    # Вес 2 (До метро): -0.8 (чем дальше метро, тем ниже цена)
    # Смещение (Bias): +10.0 (базовая стоимость за сам факт аренды коробки)
    market_weights = [1.5, -0.8]
    market_bias = 10.0

    # Создаем наш искусственный нейрон
    rent_neuron = Neuron(weights=market_weights, bias=market_bias)

    print("--- Искусственный нейрон инициализирован ---")
    print(f"Веса (w): {rent_neuron.weights}")
    print(f"Смещение (b): {rent_neuron.bias}\n")

    # Передаем на вход параметры конкретной квартиры (в масштабированном виде)
    # Квартира №1: Большая площадь (0.8), близко к метро (0.1)
    apartment_1 = [0.8, 0.1]
    prediction_1 = rent_neuron.forward(apartment_1)

    # Квартира №2: Маленькая площадь (0.2), далеко от метро (0.9)
    apartment_2 = [0.2, 0.9]
    prediction_2 = rent_neuron.forward(apartment_2)

    print("=== Результаты симуляции ===")
    print(
        f"Квартира №1 (Большая, у метро)  -> Прогноз нейрона: {prediction_1:.2f}"
    )
    print(
        f"Квартира №2 (Маленькая, далеко) -> Прогноз нейрона: {prediction_2:.2f}"
    )