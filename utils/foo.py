# #!/usr/bin/env python3
# import json
# import logging
# import sys
# import signal
# from confluent_kafka import Consumer, KafkaError
#
# # Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
# # Настройки Kafka
# KAFKA_BROKER = "localhost:9092"
# KAFKA_TOPIC = "data_collection_status"
# CONSUMER_GROUP = "simple_listener_group"
#
# # Флаг для обработки сигналов завершения
# running = True
#
#
# def signal_handler(sig, frame):
#     """Обработчик сигналов для корректного завершения"""
#     global running
#     logger.info(f"Получен сигнал завершения, выходим...")
#     running = False
#
#
# def listen_for_completion():
#     """Слушает Kafka-топик и ждет сообщение о завершении сбора данных"""
#     # Настройка обработчиков сигналов
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
#
#     # Настройка консьюмера
#     consumer_config = {
#         'bootstrap.servers': KAFKA_BROKER,
#         'group.id': CONSUMER_GROUP,
#         'auto.offset.reset': 'latest',  # Начинаем с последних сообщений
#         'enable.auto.commit': True,
#         'auto.commit.interval.ms': 5000
#     }
#
#     consumer = Consumer(consumer_config)
#     consumer.subscribe([KAFKA_TOPIC])
#
#     logger.info(f"Ожидаем сообщения о статусе сбора данных из топика {KAFKA_TOPIC}...")
#     logger.info("Нажмите Ctrl+C для выхода.")
#
#     try:
#         while running:
#             # Получаем сообщение (ждем не более 1 секунды)
#             msg = consumer.poll(1.0)
#
#             if msg is None:
#                 continue
#
#             if msg.error():
#                 if msg.error().code() == KafkaError._PARTITION_EOF:
#                     continue
#                 else:
#                     logger.error(f"Ошибка получения сообщения: {msg.error()}")
#                     continue
#
#             # Декодируем сообщение
#             try:
#                 value = msg.value().decode('utf-8')
#                 status = json.loads(value)
#
#                 # Проверяем, тестовое ли это сообщение
#                 if status.get("test", False):
#                     logger.info(f"Получено тестовое сообщение: {status.get('message', '')}")
#                     continue
#
#                 # Проверяем статус завершения
#                 if "completed" in status:
#                     if status["completed"]:
#                         # Сбор данных успешно завершен
#                         execution_time = status.get("execution_time", "неизвестно")
#                         logger.info(f"✅ СБОР ДАННЫХ УСПЕШНО ЗАВЕРШЕН!")
#                         logger.info(f"⏱️ Время выполнения: {execution_time} секунд")
#                         print("\n\n*** СБОР ДАННЫХ УСПЕШНО ЗАВЕРШЕН! ***\n\n")
#                     else:
#                         # Проверяем, есть ли ошибка
#                         if "error" in status:
#                             logger.error(f"❌ ОШИБКА в процессе сбора данных: {status['error']}")
#                             print(f"\n\n*** ОШИБКА: {status['error']} ***\n\n")
#                         else:
#                             # Это сообщение о начале сбора
#                             logger.info(f"🔄 Процесс сбора данных НАЧАТ")
#
#             except Exception as e:
#                 logger.error(f"Ошибка обработки сообщения: {e}")
#
#     except KeyboardInterrupt:
#         logger.info("Прервано пользователем")
#     finally:
#         # Закрываем консьюмер
#         consumer.close()
#         logger.info("Слушатель остановлен")
#
#
# if __name__ == "__main__":
#     logger.info("Запуск слушателя Kafka")
#     listen_for_completion()
#


