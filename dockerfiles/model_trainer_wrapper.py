#!/usr/bin/env python3
#!/usr/bin/env python3
import io
import json
import logging
import os
import signal
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

from confluent_kafka import Consumer, KafkaError
from confluent_kafka import Producer
from dotenv import load_dotenv
# Добавляем SQLAlchemy для безопасной работы с БД
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import re
# import datetime



from pydantic_models.model_result import ModelResult

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path configurations
DATAFRAMES_DIR = os.getenv("DATAFRAMES_DIR")
# SHARED_DIR = "/shared"
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
WAIT_TIMEOUT = int(os.getenv("WAIT_TIMEOUT"))  # Maximum wait time in seconds (1 hour)

DB_HOST = os.getenv("DB_HOST") # "localhost"  # Change this to your database host
DB_PORT = os.getenv("DB_PORT")  # "5432"  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")    # "postgres"  # Change to your database name
DB_USER = os.getenv("DB_USER")      # "postgres"  # Change to your username
DB_PASSWORD = os.getenv("DB_PASSWORD")    # "mysecretpassword"  # Change to your password
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER")        # "localhost:9092"  # Измените на адрес вашего Kafka-брокера
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")     # "data_collection_status"
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP")    #"model_trainer_group"  # Уникальная группа для этого потребителя
KAFKA_TOPIC_PRODUCER = os.getenv("KAFKA_TOPIC_PRODUCER")        # "model_trainer_status"  # Тема для отправки статуса

engine = create_engine(DATABASE_URL)
metadata = MetaData(schema="public")  # Указываем схему public
Session = sessionmaker(bind=engine)


def extract_metrics(text):
    """
    Извлекает метрики из текста логов модели
    """
    # Регулярные выражения для поиска метрик
    patterns = {
        'test_mse': r'===== Метрики на тестовой выборке =====.*?MSE: ([\d.]+)',
        'test_rmse': r'===== Метрики на тестовой выборке =====.*?RMSE: ([\d.]+)',
        'test_mae': r'===== Метрики на тестовой выборке =====.*?MAE: ([\d.]+)',
        'test_r2': r'===== Метрики на тестовой выборке =====.*?R²: ([\d.]+)',
        'test_mape': r'===== Метрики на тестовой выборке =====.*?MAPE: ([\d.]+)',
        'test_direction_accuracy': r'===== Метрики на тестовой выборке =====.*?Direction Accuracy: ([\d.]+)',
        'train_direction_accuracy': r'===== Метрики на обучающей выборке =====.*?Direction Accuracy: ([\d.]+)',

        # Добавляем парсинг текущей и прогнозируемой цены
        'current_price': r'Текущая цена: ([\d.]+)',
        'predicted_price': r'Прогнозируемая цена: ([\d.]+)',
        'expected_change': r'Ожидаемое изменение: ([-+]?[\d.]+)%',
        'trading_signal': r'Торговый сигнал: (\w+)'
    }

    # Извлекаем значения
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if key in ['current_price', 'predicted_price', 'expected_change']:
                metrics[key] = float(match.group(1))
            elif key == 'trading_signal':
                metrics[key] = match.group(1).strip()
            else:
                metrics[key] = float(match.group(1))
        else:
            metrics[key] = None

    return metrics


def save_result_to_db(db_name, model_name, timestamp, output_text):
    """Сохранение результатов модели в базу данных с использованием ORM"""
    try:
        # Создаем сессию
        session = Session()

        metrics = extract_metrics(output_text)

        # Создаем новую запись
        new_result = ModelResult(
            db_name=db_name,
            model_name=model_name,
            timestamp=timestamp,
            text=output_text,
            test_mse=metrics['test_mse'],
            test_rmse=metrics['test_rmse'],
            test_mae=metrics['test_mae'],
            test_r2=metrics['test_r2'],
            test_mape=metrics['test_mape'],
            test_direction_accuracy=metrics['test_direction_accuracy'],
            train_direction_accuracy=metrics['train_direction_accuracy'],
            current_price=metrics['current_price'],
            predicted_price=metrics['predicted_price'],
            expected_change=metrics['expected_change'],
            trading_signal=metrics['trading_signal']
        )

        # Добавляем и сохраняем
        session.add(new_result)
        session.commit()

        logger.info(f"Saved result for {model_name} on {db_name} to database")

        # Закрываем сессию
        session.close()

        return True

    except Exception as e:
        logger.error(f"Error saving result to database: {e}")
        return False


def check_data_collection_status():
    """Check if data collection process has completed using Kafka messaging"""
    logger.info(f"Waiting for data collection completion status from Kafka topic: {KAFKA_TOPIC}")

    # Флаг для индикации завершения или ошибки
    collection_completed = False
    collection_error = None
    execution_time = None

    # Флаг для обработки сигналов завершения и цикла ожидания
    running = True

    # Обработчик сигналов для корректного завершения
    def signal_handler(sig, frame):
        nonlocal running
        logger.info(f"Received termination signal, exiting...")
        running = False

    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Настройка консьюмера Kafka
    consumer_config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'latest',  # Получаем только новые сообщения
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 5000
    }

    consumer = Consumer(consumer_config)
    consumer.subscribe([KAFKA_TOPIC])

    # Время начала ожидания
    start_time = time.time()

    try:
        logger.info("Waiting for data collection to complete...")

        # Основной цикл ожидания сообщений
        while running and (time.time() - start_time < WAIT_TIMEOUT):
            # Получаем сообщение (ждем не более 1 секунды)
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Error receiving message: {msg.error()}")
                    continue

            # Обрабатываем полученное сообщение
            try:
                value = msg.value().decode('utf-8')
                status = json.loads(value)

                # Пропускаем тестовые сообщения
                if status.get("test", False):
                    logger.info(f"Received test message: {status.get('message', '')}")
                    continue

                # Проверяем статус завершения
                if "completed" in status:
                    if status["completed"]:
                        # Сбор данных успешно завершен
                        execution_time = status.get("execution_time", "unknown")
                        logger.info(f"✅ DATA COLLECTION SUCCESSFULLY COMPLETED!")
                        logger.info(f"⏱️ Execution time: {execution_time} seconds")

                        collection_completed = True
                        break  # Выходим из цикла, так как получили статус завершения

                    elif "error" in status:
                        # Произошла ошибка при сборе данных
                        error_msg = status.get("error", "Unknown error")
                        logger.error(f"❌ ERROR in data collection process: {error_msg}")

                        collection_error = error_msg
                        break  # Выходим из цикла, так как получили статус ошибки
                    else:
                        # Это сообщение о начале сбора данных
                        logger.info(f"🔄 Data collection process STARTED")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except Exception as e:
        logger.error(f"Unexpected error while waiting for data collection status: {e}")

    finally:
        # Закрываем консьюмер в любом случае
        consumer.close()

    # Проверяем результат ожидания
    if collection_completed:
        logger.info("Data collection has completed successfully!")
        return True
    elif collection_error:
        logger.error(f"Data collection failed with error: {collection_error}")
        return False
    elif (time.time() - start_time) >= WAIT_TIMEOUT:
        logger.error(f"Timeout waiting for data collection after {WAIT_TIMEOUT} seconds")
        return False
    else:
        logger.error("Data collection process monitoring interrupted")
        return False


def get_available_db_files():
    """Get all available database files from the all_dfs schema in PostgreSQL"""
    try:
        session = Session()

        # Query to list all tables in the all_dfs schema that end with .db
        # We're using raw SQL with text() since this is a database metadata query
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'all_dfs' 
        """)

        result = session.execute(query)
        db_files = [row[0] for row in result]

        session.close()

        logger.info(f"Found {len(db_files)} database tables with .db suffix in all_dfs schema")
        return db_files
    except Exception as e:
        logger.error(f"Error fetching database files from schema: {e}")
        return []

def run_model(model_name, db_path):
    """Run a specific model on a database file and capture its output"""
    logger.info(f"Running {model_name} on {db_path}")

    try:
        # Import the module
        if model_name == "ridge":
            from models.ridge import main
        elif model_name == "rf_classifier":
            from models.rf_classifier import main
        elif model_name == "arima":
            from models.arima import predict_stock_with_arima as main
        elif model_name == "lstm_model":
            from models.lstm_model import main
        elif model_name == "random_forest_regression_model":
            from models.random_forest_regression_model import main
        elif model_name == "xgboost_model":
            from models.xgboost_model import main
        elif model_name == "lightgbm_model":
            from models.lightgbm_model import main
        elif model_name == "tcn_model":
            from models.tcn_model import main
        elif model_name == "prophet_model":
            from models.prophet_model import main
        # elif model_name == "rdpg_lstm_model":
        #     from models.rdpg_lstm_model import main
        elif model_name == "cat_boost_model":
            from models.cat_boost_model import main
        else:
            logger.error(f"Unknown model: {model_name}")
            return None

        # Capture standard output
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # Handle arima specially since it's an async function
                if model_name == "arima":
                    import asyncio
                    asyncio.run(main(db_path))
                else:
                    main(db_path)
            except Exception as e:
                print(f"Error executing {model_name} on {db_path}: {str(e)}")

        # Get the output
        output = f.getvalue()
        return output

    except ImportError as e:
        logger.error(f"Failed to import {model_name}: {e}")
        return f"Error: Could not import {model_name}. {str(e)}"
    except Exception as e:
        logger.error(f"Error running {model_name}: {e}")
        return f"Error running {model_name}: {str(e)}"


def train_models():
    """Train all models on available data"""
    # Create output directory if it doesn't exist
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all database files
    db_files = get_available_db_files()
    if not db_files:
        logger.error("No database files found to train models on!")
        return

    # Define models to run
    models = [
        "random_forest_regression_model",
        "ridge",
        "rf_classifier",
        "xgboost_model",
        # "lstm_model",
        # "arima",
        # "tcn_model",
        "prophet_model",
        "cat_boost_model"
    ]

    # # Check if lightgbm and tcn are available
    # try:
    #     import models.lightgbm_model
    #     models.append("lightgbm_model")
    # except ImportError:
    #     logger.warning("lightgbm_model not available, skipping")
    #
    # try:
    #     import models.tcn_model
    #     models.append("tcn_model")
    # except ImportError:
    #     logger.warning("tcn_model not available, skipping")

    # Log start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a summary file
    summary_file = f"{OUTPUT_DIR}/training_summary_{timestamp}.txt"
    # with open(summary_file, 'w') as summary:
    # summary.write(f"Model Training Summary - {datetime.now()}\n")
    # summary.write("=" * 80 + "\n\n")

    # Process each database file
    for db_path in db_files:
        db_name = os.path.basename(db_path)
        # summary.write(f"\nProcessing database: {db_name}\n")
        # summary.write("-" * 80 + "\n")

        # Run each model on the database
        for model_name in models:
            model_start = time.time()
            logger.info(f"Starting {model_name} on {db_name}")

            # Create output file for this model run
            output_file = f"{OUTPUT_DIR}/{db_name}_{model_name}_{timestamp}.txt"

            # Run the model and capture output
            output = run_model(model_name, db_path)

            # Save output to file
            if output:
                save_result_to_db(db_name, model_name, timestamp, output)


                # Выпилил, потому что теперь сохраняем в бд
                # with open(output_file, 'w') as f:
                #     f.write(output)

                model_end = time.time()
                duration = model_end - model_start

                # summary.write(f"{model_name}: Completed in {duration:.2f} seconds\n")
                logger.info(f"Completed {model_name} on {db_name} in {duration:.2f} seconds")

            else:
                # summary.write(f"{model_name}: Failed\n")
                logger.error(f"Failed to run {model_name} on {db_name}")
                send_status_to_kafka({
                    "completed": False,
                    "timestamp": time.time()
                })

        # Log total execution time
        end_time = time.time()
        total_duration = end_time - start_time
        # summary.write(f"\nTotal execution time: {total_duration:.2f} seconds\n")
        logger.info(f"All models trained in {total_duration:.2f} seconds")

        # УБИРАЮ, Т.К. НАСТРОИЛ КАФКУ
        # with open(f"{SHARED_DIR}/training_completed", 'w') as f:
        #     f.write(f"DONE")

        send_status_to_kafka({
            "completed": True,
            "timestamp": time.time()
        })

def create_kafka_producer():
    """Создает и возвращает продюсер Kafka"""
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'client.id': 'model_trainer_producer'
    }
    logger.info(f"Подключение к Kafka брокеру: {KAFKA_BROKER}")
    return Producer(conf)

def send_status_to_kafka(status):
    """Отправляет статус в Kafka"""
    producer = create_kafka_producer()

    try:
        # Сериализуем JSON в строку
        status_json = json.dumps(status)

        # Отправляем сообщение
        producer.produce(
            KAFKA_TOPIC_PRODUCER,
            key="trainer_status",
            value=status_json.encode('utf-8'),  # Явно кодируем в bytes
            callback=delivery_report
        )

        # Ждем отправки всех сообщений
        producer.flush(timeout=10)  # Добавляем таймаут для диагностики

        logger.info(f"Статус отправлен в Kafka: {status}")
    except Exception as e:
        logger.error(f"Ошибка отправки в Kafka: {str(e)}")


def delivery_report(err, msg):
    """Callback для подтверждения доставки сообщения"""
    if err is not None:
        logger.error(f'Ошибка доставки сообщения: {err}')
    else:
        logger.info(f'Сообщение доставлено в {msg.topic()} [{msg.partition()}]')


def main():
    logger.info("Starting model training process")

    # Wait for data collection to complete
    if not check_data_collection_status():
        logger.error("Data collection process did not complete successfully")
        return
        # sys.exit(1)

    # Train models on the collected data
    train_models()

    logger.info("Model training process completed")


if __name__ == "__main__":
    while True:
        current_time = datetime.now()
        if current_time.hour == 23:
            main()

        logger.info("ЗАСЫПАЕМ НА НЕСКОЛЬКО ЧАСОВ В ОЖИДАНИИ НОВОГО ТОРГОВОГО ИНТЕРВАЛА ДЛЯ ОБУЧЕНИЯ")
        time.sleep(3600)