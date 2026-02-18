import re
import asyncio


async def calculate_model_score(metrics: dict) -> float:
    """
    Вычисляет комплексную оценку модели на основе метрик и анализа последних предсказаний

    Args:
        metrics: словарь с метриками модели

    Returns:
        float: итоговая оценка (чем больше, тем лучше)
    """

    # Коэффициенты важности для каждой метрики
    weights = {
        'test_r2': 0.25,
        'test_rmse': -0.20,
        'test_mae': -0.12,
        'test_mape': -0.08,
        'test_direction_accuracy': 0.15,
        'train_direction_accuracy': 0.08,
        'prediction_accuracy': 0.12  # Новый вес для точности последних предсказаний
    }

    # Функция для безопасного получения числового значения
    def safe_get_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # Функция для анализа последних предсказаний из текста
    def analyze_prediction_text(text: str) -> float:
        """
        Анализирует текст с последними предсказаниями и возвращает оценку точности

        Returns:
            float: оценка от 0 до 1 (1 = отличная точность)
        """
        if not text or not isinstance(text, str):
            return 0.0

        try:
            # Ищем все ошибки в процентах
            error_pattern = r'Ошибка: (-?\d+\.?\d*)%'
            errors = re.findall(error_pattern, text)

            if not errors:
                return 0.0

            # Конвертируем в числа, игнорируя 'nan'
            valid_errors = []
            for error in errors:
                try:
                    error_val = abs(float(error))  # Берем абсолютное значение
                    if not (error_val != error_val):  # Проверка на NaN
                        valid_errors.append(error_val)
                except ValueError:
                    continue

            if not valid_errors:
                return 0.0

            # Вычисляем среднюю абсолютную ошибку
            avg_error = sum(valid_errors) / len(valid_errors)

            # Преобразуем в оценку (чем меньше ошибка, тем лучше)
            if avg_error <= 1.0:  # Ошибка <= 1%
                return 1.0
            elif avg_error <= 2.0:  # Ошибка <= 2%
                return 0.8
            elif avg_error <= 3.0:  # Ошибка <= 3%
                return 0.6
            elif avg_error <= 5.0:  # Ошибка <= 5%
                return 0.4
            elif avg_error <= 10.0:  # Ошибка <= 10%
                return 0.2
            else:
                return 0.1  # Ошибка > 10%

        except Exception as e:
            print(f"Ошибка при анализе текста предсказаний: {e}")
            return 0.0

    # Нормализация метрик с проверкой на None
    async def normalize_metrics(metrics):
        normalized = {}

        # R² - безопасное получение значения
        r2_value = safe_get_float(metrics.get('test_r2'), 0.0)
        normalized['test_r2'] = max(0, min(1, r2_value))

        # RMSE - проверяем на None и отрицательные значения
        rmse_value = safe_get_float(metrics.get('test_rmse'), 1.0)
        if rmse_value <= 0:
            rmse_value = 1.0
        normalized['test_rmse'] = 1 / (1 + rmse_value)

        # MAE - аналогично RMSE
        mae_value = safe_get_float(metrics.get('test_mae'), 1.0)
        if mae_value <= 0:
            mae_value = 1.0
        normalized['test_mae'] = 1 / (1 + mae_value)

        # MAPE - процентная ошибка
        mape_value = safe_get_float(metrics.get('test_mape'), 100.0)
        if mape_value <= 0:
            mape_value = 100.0
        normalized['test_mape'] = 1 / (1 + mape_value / 100)

        # Direction accuracy - приводим к [0, 1]
        test_dir_acc = safe_get_float(metrics.get('test_direction_accuracy'), 0.0)
        normalized['test_direction_accuracy'] = max(0, min(100, test_dir_acc)) / 100

        train_dir_acc = safe_get_float(metrics.get('train_direction_accuracy'), 0.0)
        normalized['train_direction_accuracy'] = max(0, min(100, train_dir_acc)) / 100

        # Анализ точности последних предсказаний из текста
        text_content = metrics.get('text', '')
        normalized['prediction_accuracy'] = analyze_prediction_text(text_content)

        return normalized

    try:
        # Нормализуем метрики
        norm_metrics = await normalize_metrics(metrics)

        # Вычисляем взвешенную сумму
        score = 0
        total_weights = 0

        for metric, weight in weights.items():
            if metric in norm_metrics and norm_metrics[metric] is not None:
                score += norm_metrics[metric] * abs(weight)
                total_weights += abs(weight)

        # Если нет валидных метрик, возвращаем 0
        if total_weights == 0:
            return 0.0

        # Нормализуем по сумме весов и приводим к шкале 0-100
        final_score = (score / total_weights) * 100
        return round(max(0, min(100, final_score)), 2)

    except Exception as e:
        print(f"Ошибка в calculate_model_score: {e}")
        return 0.0


# Дополнительная функция для проверки валидности модели
def is_valid_model(model_data: dict) -> bool:
    """Проверяет, есть ли у модели хотя бы одна валидная метрика"""
    required_metrics = ['test_r2', 'test_rmse', 'test_mae', 'test_direction_accuracy']

    for metric in required_metrics:
        value = model_data.get(metric)
        if value is not None:
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                continue

    return False


# Функция для детального анализа предсказаний (опционально)
def get_prediction_details(text: str) -> dict:
    """
    Возвращает детальную информацию о последних предсказаниях
    """
    if not text or not isinstance(text, str):
        return {}

    try:
        error_pattern = r'Ошибка: (-?\d+\.?\d*)%'
        errors = re.findall(error_pattern, text)

        valid_errors = []
        for error in errors:
            try:
                error_val = float(error)
                if not (error_val != error_val):  # Проверка на NaN
                    valid_errors.append(abs(error_val))
            except ValueError:
                continue

        if not valid_errors:
            return {}

        return {
            "avg_error": round(sum(valid_errors) / len(valid_errors), 2),
            "max_error": round(max(valid_errors), 2),
            "min_error": round(min(valid_errors), 2),
            "predictions_count": len(valid_errors),
            "good_predictions": len([e for e in valid_errors if e <= 5.0])  # Ошибка <= 5%
        }

    except Exception:
        return {}

# Пример использования:
# metrics_example = {
#     "test_mse": 0.0469,
#     "test_rmse": 0.2166,
#     "test_mae": 0.1501,
#     "test_r2": 0.9904,
#     "test_mape": 0.1583,
#     "test_direction_accuracy": 65.4676,
#     "train_direction_accuracy": 49.0161,
# }
#
# score = asyncio.run(calculate_model_score(metrics_example))
# print(f"Оценка модели: {score}")  # Выведет что-то вроде: 82.45