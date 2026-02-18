from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, MetaData
from pydantic import BaseModel
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ModelResult(Base):
    """Модель для хранения результатов работы моделей"""
    __tablename__ = 'model_results'
    __table_args__ = {'schema': 'public'}  # Явно указываем схему

    id = Column(Integer, primary_key=True)
    db_name = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    text = Column(Text, nullable=False)

    # Метрики тестовой выборки
    test_mse = Column(Float, nullable=True)
    test_rmse = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    test_mape = Column(Float, nullable=True)
    test_direction_accuracy = Column(Float, nullable=True)

    # Метрики обучающей выборки
    train_direction_accuracy = Column(Float, nullable=True)

    # Прогнозные данные
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    expected_change = Column(Float, nullable=True)
    trading_signal = Column(String(10), nullable=True)

    def __repr__(self):
        return f"<ModelResult(model_name='{self.model_name}', db_name='{self.db_name}', test_r2={self.test_r2})>"