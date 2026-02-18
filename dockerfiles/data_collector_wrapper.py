#!/usr/bin/env python3
import json
import logging
import os
import time
from confluent_kafka import Producer
from all_dfs_to_db import main as collect_data
from dotenv import load_dotenv
import datetime

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka configuration
# Для локального запуска с Kafka в Docker используем localhost
KAFKA_BROKER = os.getenv("KAFKA_BROKER")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")  # Тема для отправки статуса


def delivery_report(err, msg):
    """Callback для подтверждения доставки сообщения"""
    if err is not None:
        logger.error(f'Ошибка доставки сообщения: {err}')
    else:
        logger.info(f'Сообщение доставлено в {msg.topic()} [{msg.partition()}]')


def create_kafka_producer():
    """Создает и возвращает продюсер Kafka"""
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'client.id': 'data_collector_producer'
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
            KAFKA_TOPIC,
            key="collection_status",
            value=status_json.encode('utf-8'),  # Явно кодируем в bytes
            callback=delivery_report
        )

        # Ждем отправки всех сообщений
        producer.flush(timeout=10)  # Добавляем таймаут для диагностики

        logger.info(f"Статус отправлен в Kafka: {status}")
    except Exception as e:
        logger.error(f"Ошибка отправки в Kafka: {str(e)}")


def run_collection_process():
    """Запускает процесс сбора данных и отправляет его статус в Kafka"""
    try:
        # Создаем директории, если их нет
        data_dir = "/Users/vladimirmakeev/PycharmProjects/arima/app/dataframes"
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Используем директорию данных: {data_dir}")


        # Пробный статус для проверки соединения
        logger.info("Отправка тестового сообщения в Kafka...")
        send_status_to_kafka({
            "test": True,
            "message": "Тестовое сообщение",
            "timestamp": time.time()
        })

        # Отправляем начальный статус
        logger.info("Отправка начального статуса...")
        send_status_to_kafka({
            "completed": False,
            "timestamp": time.time()
        })

        logger.info("Запуск процесса сбора данных")
        start_time = time.time()

        # Запускаем сбор данных
        collect_data()

        # Вычисляем время выполнения
        end_time = time.time()
        execution_time = end_time - start_time

        # Отправляем статус завершения
        status = {
            "completed": True,
            "timestamp": time.time(),
            "execution_time": execution_time
        }

        logger.info("Отправка финального статуса...")
        send_status_to_kafka(status)

        logger.info(f"Процесс сбора данных завершен за {execution_time:.2f} секунд")

    except Exception as e:
        logger.error(f"Ошибка в процессе сбора данных: {str(e)}")
        # Отправляем статус с ошибкой
        send_status_to_kafka({
            "completed": False,
            "error": str(e),
            "timestamp": time.time()
        })


if __name__ == "__main__":

    while True:
        # send_status_to_kafka({
        #     "completed": True,
        #     "timestamp": time.time()
        # })

        # Способ 1: Получить текущее время и проверить час
        current_time = datetime.datetime.now()
        if current_time.hour == 23:
            run_collection_process()
        else:
            logger.info("ЗАСЫПАЕМ НА НЕСКОЛЬКО ЧАСОВ В ОЖИДАНИИ НОВОГО ТОРГОВОГО ИНТЕРВАЛА")
            time.sleep(3600)