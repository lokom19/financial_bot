"""
Система грейдов для оценки надёжности предсказаний.

Грейды от A до F оценивают качество предсказания на основе:
- Консенсус моделей (сколько моделей согласны с сигналом)
- R² (качество модели)
- Direction Accuracy (точность направления)
- Win Rate (процент прибыльных сделок на бэктесте)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class GradeThresholds:
    """Пороговые значения для грейдов."""
    # Консенсус моделей (%)
    consensus_a: float = 80.0  # A: >= 80%
    consensus_b: float = 60.0  # B: >= 60%
    consensus_c: float = 40.0  # C: >= 40%
    consensus_d: float = 20.0  # D: >= 20%
    # F: < 20%

    # R² (коэффициент детерминации)
    r2_a: float = 0.7   # A: >= 0.7
    r2_b: float = 0.5   # B: >= 0.5
    r2_c: float = 0.3   # C: >= 0.3
    r2_d: float = 0.1   # D: >= 0.1
    # F: < 0.1

    # Direction Accuracy (%)
    direction_a: float = 65.0  # A: >= 65%
    direction_b: float = 55.0  # B: >= 55%
    direction_c: float = 50.0  # C: >= 50%
    direction_d: float = 45.0  # D: >= 45%
    # F: < 45%

    # Win Rate (%)
    winrate_a: float = 60.0  # A: >= 60%
    winrate_b: float = 55.0  # B: >= 55%
    winrate_c: float = 50.0  # C: >= 50%
    winrate_d: float = 45.0  # D: >= 45%
    # F: < 45%


# Веса для расчёта общего грейда
GRADE_WEIGHTS = {
    'consensus': 0.35,    # Консенсус моделей - самый важный
    'r2': 0.25,           # R² - качество модели
    'direction': 0.25,    # Direction Accuracy - точность направления
    'winrate': 0.15,      # Win Rate - исторический результат
}

# Преобразование числового грейда в буквенный
GRADE_LETTERS = {
    5: 'A',
    4: 'B',
    3: 'C',
    2: 'D',
    1: 'F',
}

# Цвета для грейдов
GRADE_COLORS = {
    'A': '#28a745',  # Зелёный
    'B': '#5cb85c',  # Светло-зелёный
    'C': '#f0ad4e',  # Оранжевый
    'D': '#d9534f',  # Красный
    'F': '#721c24',  # Тёмно-красный
}

# Описания грейдов
GRADE_DESCRIPTIONS = {
    'A': 'Отличный сигнал - высокая надёжность',
    'B': 'Хороший сигнал - выше среднего',
    'C': 'Средний сигнал - умеренная надёжность',
    'D': 'Слабый сигнал - низкая надёжность',
    'F': 'Ненадёжный сигнал - не рекомендуется',
}


def get_metric_grade(value: Optional[float], thresholds: Tuple[float, float, float, float]) -> int:
    """
    Получить числовой грейд для метрики.

    Args:
        value: Значение метрики
        thresholds: Пороги (A, B, C, D) в порядке убывания

    Returns:
        Числовой грейд: 5=A, 4=B, 3=C, 2=D, 1=F
    """
    if value is None:
        return 1  # F для отсутствующих данных

    a, b, c, d = thresholds

    if value >= a:
        return 5  # A
    elif value >= b:
        return 4  # B
    elif value >= c:
        return 3  # C
    elif value >= d:
        return 2  # D
    else:
        return 1  # F


def calculate_signal_grade(
    consensus_pct: Optional[float] = None,
    r2: Optional[float] = None,
    direction_accuracy: Optional[float] = None,
    win_rate: Optional[float] = None,
    thresholds: Optional[GradeThresholds] = None
) -> dict:
    """
    Рассчитать грейд для торгового сигнала.

    Args:
        consensus_pct: Процент консенсуса моделей (0-100)
        r2: R² модели (0-1)
        direction_accuracy: Точность направления (0-100)
        win_rate: Win rate (0-100)
        thresholds: Пользовательские пороги (опционально)

    Returns:
        Словарь с грейдами по каждой метрике и общим грейдом
    """
    if thresholds is None:
        thresholds = GradeThresholds()

    # Грейды по каждой метрике
    grades = {
        'consensus': get_metric_grade(
            consensus_pct,
            (thresholds.consensus_a, thresholds.consensus_b,
             thresholds.consensus_c, thresholds.consensus_d)
        ),
        'r2': get_metric_grade(
            r2,
            (thresholds.r2_a, thresholds.r2_b, thresholds.r2_c, thresholds.r2_d)
        ),
        'direction': get_metric_grade(
            direction_accuracy,
            (thresholds.direction_a, thresholds.direction_b,
             thresholds.direction_c, thresholds.direction_d)
        ),
        'winrate': get_metric_grade(
            win_rate,
            (thresholds.winrate_a, thresholds.winrate_b,
             thresholds.winrate_c, thresholds.winrate_d)
        ),
    }

    # Взвешенный средний грейд
    total_weight = 0
    weighted_sum = 0

    for metric, grade in grades.items():
        weight = GRADE_WEIGHTS[metric]
        # Пропускаем метрики без данных (grade=1 означает F или нет данных)
        # Но если есть хотя бы consensus, учитываем
        if metric == 'consensus' or grade > 1:
            weighted_sum += grade * weight
            total_weight += weight

    if total_weight > 0:
        avg_grade = weighted_sum / total_weight
    else:
        avg_grade = 1  # F если нет данных

    # Округляем до ближайшего целого грейда
    overall_grade = round(avg_grade)
    overall_grade = max(1, min(5, overall_grade))  # Ограничиваем 1-5

    # Конвертируем в буквы
    result = {
        'overall': GRADE_LETTERS[overall_grade],
        'overall_numeric': overall_grade,
        'consensus_grade': GRADE_LETTERS[grades['consensus']],
        'r2_grade': GRADE_LETTERS[grades['r2']],
        'direction_grade': GRADE_LETTERS[grades['direction']],
        'winrate_grade': GRADE_LETTERS[grades['winrate']],
        'color': GRADE_COLORS[GRADE_LETTERS[overall_grade]],
        'description': GRADE_DESCRIPTIONS[GRADE_LETTERS[overall_grade]],
    }

    return result


def add_grades_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить колонку с грейдами к DataFrame с сигналами.

    Args:
        df: DataFrame с колонками consensus_pct, avg_r2, direction_accuracy, win_rate

    Returns:
        DataFrame с добавленными колонками grade и grade_color
    """
    if df.empty:
        return df

    grades = []
    colors = []
    descriptions = []

    for _, row in df.iterrows():
        grade_info = calculate_signal_grade(
            consensus_pct=row.get('consensus_pct'),
            r2=row.get('avg_r2'),
            direction_accuracy=row.get('direction_accuracy'),
            win_rate=row.get('win_rate'),
        )
        grades.append(grade_info['overall'])
        colors.append(grade_info['color'])
        descriptions.append(grade_info['description'])

    df = df.copy()
    df['grade'] = grades
    df['grade_color'] = colors
    df['grade_description'] = descriptions

    return df


def get_grade_legend() -> str:
    """
    Получить легенду грейдов в формате Markdown.
    """
    legend = """
### Система грейдов

| Грейд | Надёжность | Описание |
|:-----:|:----------:|:---------|
| **A** | Отличная | Консенсус ≥80%, R² ≥0.7, Direction ≥65% |
| **B** | Хорошая | Консенсус ≥60%, R² ≥0.5, Direction ≥55% |
| **C** | Средняя | Консенсус ≥40%, R² ≥0.3, Direction ≥50% |
| **D** | Низкая | Консенсус ≥20%, R² ≥0.1, Direction ≥45% |
| **F** | Очень низкая | Ниже пороговых значений |

**Веса метрик:**
- Консенсус моделей: 35%
- R² (качество модели): 25%
- Direction Accuracy: 25%
- Win Rate (бэктест): 15%
"""
    return legend
