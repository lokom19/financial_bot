from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ModelResult(Base):
    """Модель для хранения результатов работы ML моделей"""
    __tablename__ = 'model_results'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True)
    db_name = Column(String(255), nullable=False)  # FIGI инструмента
    ticker_name = Column(String(50), nullable=True)  # Человекочитаемое имя (SBER, YNDX)
    model_name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    text = Column(Text, nullable=False)  # Полный вывод модели

    # Информация о данных
    train_samples = Column(Integer, nullable=True)  # Кол-во записей в train
    test_samples = Column(Integer, nullable=True)  # Кол-во записей в test
    data_start_date = Column(Date, nullable=True)  # Начало периода данных
    data_end_date = Column(Date, nullable=True)  # Конец периода данных

    # Метрики тестовой выборки
    test_mse = Column(Float, nullable=True)
    test_rmse = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    test_mape = Column(Float, nullable=True)
    test_direction_accuracy = Column(Float, nullable=True)

    # Метрики обучающей выборки
    train_mse = Column(Float, nullable=True)
    train_rmse = Column(Float, nullable=True)
    train_mae = Column(Float, nullable=True)
    train_r2 = Column(Float, nullable=True)
    train_direction_accuracy = Column(Float, nullable=True)

    # Прогнозные данные
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    expected_change = Column(Float, nullable=True)  # В процентах
    prediction_std = Column(Float, nullable=True)  # Стандартное отклонение прогноза
    trading_signal = Column(String(10), nullable=True)  # BUY/SELL/HOLD/NEUTRAL

    # Торговая эффективность (бэктест)
    total_trades = Column(Integer, nullable=True)
    profitable_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)  # profitable_trades / total_trades * 100
    profit_factor = Column(Float, nullable=True)  # sum(profits) / sum(losses)
    cumulative_return = Column(Float, nullable=True)  # Общая доходность в %

    # LLM-валидация сигнала (ночной пайплайн)
    llm_signal = Column(String(20), nullable=True)  # AGREE / DISAGREE / UNAVAILABLE
    llm_reasoning = Column(Text, nullable=True)     # Финальное обоснование (после авто-коррекции)
    llm_processed_at = Column(DateTime, nullable=True)
    # Оригинал ответа LLM до нашей пост-обработки — чтобы пользователь мог увидеть,
    # что реально сказала модель, когда наша rule-based-подстраховка её переопределила.
    llm_raw_verdict = Column(String(20), nullable=True)
    llm_raw_response = Column(Text, nullable=True)

    def __repr__(self):
        return f"<ModelResult(model='{self.model_name}', ticker='{self.ticker_name or self.db_name}', r2={self.test_r2})>"


class TickerReport(Base):
    """
    Консолидированный AI-отчёт по тикеру (один на (тикер, день)).

    LLM агрегирует прогнозы всех моделей + TA + (на будущее) новости
    и выносит ОДИН вердикт BUY/SELL/HOLD с целевыми ценами.
    Точность этих вердиктов отслеживается отдельно от per-моделной статистики.
    """
    __tablename__ = "ticker_reports"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    figi = Column(String(64), nullable=False, index=True)
    ticker = Column(String(20), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)  # когда отчёт сгенерирован
    data_date = Column(Date, nullable=True)        # на какую дату данные
    prediction_date = Column(Date, nullable=True)  # на какой день прогноз

    # Вход (текущая цена на момент отчёта)
    current_price = Column(Float, nullable=True)

    # Вердикт LLM
    verdict = Column(String(20), nullable=True)       # BUY / SELL / HOLD / UNKNOWN
    confidence = Column(String(20), nullable=True)    # высокая / средняя / низкая
    entry_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Снэпшоты входных данных (JSON-строки)
    sections_json = Column(Text, nullable=True)         # разделы отчёта
    models_snapshot_json = Column(Text, nullable=True)  # модели, что кормили отчёт
    ta_snapshot_json = Column(Text, nullable=True)      # значения индикаторов

    # Кеш LLM-ответов (чтобы не дёргать API при каждом нажатии кнопки)
    ta_summary_json = Column(Text, nullable=True)       # БЫЧИЙ/МЕДВЕЖИЙ + summary
    ta_explanation_text = Column(Text, nullable=True)   # развёрнутое TA-объяснение

    # Факт (заполняется на следующий день после prediction_date)
    actual_close = Column(Float, nullable=True)
    actual_high = Column(Float, nullable=True)
    actual_low = Column(Float, nullable=True)
    actual_resolved_at = Column(DateTime, nullable=True)
    correct_direction = Column(Boolean, nullable=True)
    target_hit = Column(Boolean, nullable=True)
    stop_hit = Column(Boolean, nullable=True)

    def __repr__(self):
        return (
            f"<TickerReport(ticker='{self.ticker}', "
            f"date={self.data_date}, verdict={self.verdict})>"
        )