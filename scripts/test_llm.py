#!/usr/bin/env python3
"""
Быстрая проверка LLM-интеграции.

Использование:
    python scripts/test_llm.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

print("=== Проверка переменных окружения ===")
groq_key = os.getenv("GROQ_API_KEY", "")
print(f"GROQ_API_KEY: {'(установлен)' if groq_key else '(НЕ установлен)'} длина={len(groq_key)}")
print(f"GROQ_MODEL:   {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')} (default)")
print(f"LLM_URL:      {os.getenv('LLM_URL', 'http://localhost:11434/api/chat')}")
print(f"LLM_MODEL:    {os.getenv('LLM_MODEL', 'llama3')}")
print()

print("=== Тестовый запрос к LLM ===")
from services.llm_service import analyze_signal, get_last_errors

result = analyze_signal(
    ticker="SBER",
    current_price=287.5,
    trading_signal="BUY",
    models_data=[
        {"model_name": "ridge", "signal": "BUY", "r2": 0.92, "direction_accuracy": 65},
        {"model_name": "xgboost", "signal": "BUY", "r2": 0.88, "direction_accuracy": 62},
    ],
    r2_avg=0.90,
    direction_accuracy_avg=63.5,
)

print(f"Ответ: {result['answer']}")
print(f"Объяснение: {result['explanation']}")
print()
print("=== Подробности ошибок (если были) ===")
errs = get_last_errors()
for provider, err in errs.items():
    print(f"  {provider}: {err or '(ок)'}")
