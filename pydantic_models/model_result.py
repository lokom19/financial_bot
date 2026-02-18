from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Date
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

    def __repr__(self):
        return f"<ModelResult(model='{self.model_name}', ticker='{self.ticker_name or self.db_name}', r2={self.test_r2})>"