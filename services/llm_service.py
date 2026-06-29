"""
LLM integration service.

Supports:
- Groq API (free tier): llama3-70b, mixtral-8x7b
- Ollama (local): any model
"""
import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# LLM system prompt
SYSTEM_PROMPT = """Ты — рецензент качества ML моделей.

ЗАДАЧА:
Тебе дают ОДНУ конкретную ML модель и её метрики. Оцени КАЧЕСТВО
ЕЁ РАБОТЫ — не рынок, не новости, не TA. Сделай вывод: можно ли
доверять её сигналу.

МЕТРИКИ (по приоритету):
1. 🔥 LIVE Direction last30d — РЕАЛЬНАЯ свежая точность модели за
   последние ~30 прогнозов в проде. Самая важная метрика. Если
   recent_samples_30d < 5 — статистики мало, опирайся на исторические.
2. Direction Accuracy на тесте — % правильно угаданных направлений
3. Win Rate — % прибыльных баров в бэктесте
4. Profit Factor — отношение прибыли к убыткам (>1 = прибыльная)
5. Cumulative Return — общая доходность бэктеста
6. test_samples — размер теста (<50 → доверие к Direction ниже)

R² игнорируй — для returns близка к 0, не показатель качества.

ПРАВИЛА (по убыванию приоритета):
- Если LIVE есть (recent_samples_30d >= 5):
    * LIVE >= 55% → AGREE
    * LIVE < 45% → DISAGREE
    * 45-55% → опирайся на Direction Accuracy и Profit Factor
- Если LIVE нет/мало (recent_samples_30d < 5):
    * Direction >= 55% AND Profit Factor >= 1.0 → AGREE
    * Direction < 50% OR Profit Factor < 0.8 → DISAGREE
    * Иначе → DISAGREE (осторожность)

ПРИМЕРЫ ПРАВИЛЬНОЙ ЛОГИКИ:
- Direction = 59.3% → 59.3 > 55, выше порога → AGREE
- Direction = 52.0% → 52 < 55, ниже порога → DISAGREE (либо смотри Profit Factor)
- Direction = 55.0% → ровно 55, на пороге → AGREE
- Direction = 44.8% → 44.8 < 55, ниже порога → DISAGREE

⚠️ САМОПРОВЕРКА:
Перед тем как написать вердикт — сравни числа.
"X% выше Y%" означает X > Y. "X% ниже Y%" означает X < Y.
Не путай больше/меньше!

ФОРМАТ ОТВЕТА (строго):
Строка 1: AGREE или DISAGREE
Строка 2-3: 1-2 коротких предложения, ССЫЛАЯСЬ НА КОНКРЕТНЫЕ ЦИФРЫ
(приоритетно LIVE, затем Direction, затем Profit Factor).
Не используй markdown."""


_LAST_GROQ_ERROR: Optional[str] = None
_LAST_MISTRAL_ERROR: Optional[str] = None
_LAST_OLLAMA_ERROR: Optional[str] = None


def get_last_errors() -> dict:
    return {
        "groq": _LAST_GROQ_ERROR,
        "mistral": _LAST_MISTRAL_ERROR,
        "ollama": _LAST_OLLAMA_ERROR,
    }


def _get_mistral_response(user_message: str, system_prompt: Optional[str] = None,
                          max_tokens: int = 250) -> Optional[str]:
    """
    Mistral AI (https://mistral.ai) — бесплатный tier с щедрыми лимитами.
    Использует OpenAI-compatible API.

    Регистрация → https://console.mistral.ai/api-keys/
    Добавь MISTRAL_API_KEY в .env. По умолчанию модель `mistral-small-latest`
    (~22B параметров — заметно умнее 8B llama).
    """
    global _LAST_MISTRAL_ERROR
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        _LAST_MISTRAL_ERROR = "MISTRAL_API_KEY не задан"
        return None

    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip().strip('"').strip("'")
    prompt = system_prompt or SYSTEM_PROMPT

    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            _LAST_MISTRAL_ERROR = None
            return resp.json()["choices"][0]["message"]["content"].strip()
        _LAST_MISTRAL_ERROR = f"HTTP {resp.status_code}: {resp.text[:150]}"
    except Exception as e:
        _LAST_MISTRAL_ERROR = f"{type(e).__name__}: {e}"
        logger.error("Mistral error: %s", e)
    return None


def _get_groq_response(user_message: str) -> Optional[str]:
    global _LAST_GROQ_ERROR
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        _LAST_GROQ_ERROR = "GROQ_API_KEY не задан"
        return None

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip().strip('"').strip("'")

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        _LAST_GROQ_ERROR = None
        return completion.choices[0].message.content.strip()
    except Exception as e:
        _LAST_GROQ_ERROR = f"{type(e).__name__}: {e}"
        logger.error("Groq API error (model=%s): %s", model, e)
        return None


def _get_ollama_response(user_message: str) -> Optional[str]:
    global _LAST_OLLAMA_ERROR
    url = os.getenv("LLM_URL", "http://localhost:11434/api/chat")
    model = os.getenv("LLM_MODEL", "llama3")
    timeout = int(os.getenv("LLM_TIMEOUT", "30"))

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            _LAST_OLLAMA_ERROR = None
            return resp.json()["message"]["content"].strip()
        _LAST_OLLAMA_ERROR = f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        _LAST_OLLAMA_ERROR = f"{type(e).__name__}: {e}"
        logger.error("Ollama error: %s", e)
    return None


def analyze_signal(
    ticker: str,
    current_price: float,
    trading_signal: str,
    models_data: list,
    r2_avg: float,
    direction_accuracy_avg: float,
    ta_indicators: Optional[dict] = None,
) -> dict:
    """
    Query LLM to analyze a trading signal.

    Args:
        ticker: Stock ticker symbol
        current_price: Current price
        trading_signal: Overall consensus signal (BUY/SELL/HOLD)
        models_data: List of dicts with model results
        r2_avg: Average R² across models
        direction_accuracy_avg: Average direction accuracy

    Returns:
        Dict with 'answer' (AGREE/DISAGREE/UNAVAILABLE) and 'explanation'
    """
    # Build model summary
    model_lines = []
    for m in models_data:
        name = m.get("model_name", "?")
        sig = m.get("signal", "?")
        r2 = m.get("r2") or 0
        da = m.get("direction_accuracy") or 50
        model_lines.append(f"  {name}: {sig}, R²={r2:.2f}, Direction={da:.0f}%")

    models_summary = "\n".join(model_lines) if model_lines else "  Нет данных"

    # Технические индикаторы (если переданы)
    ta_block = ""
    if ta_indicators:
        rsi = ta_indicators.get("rsi")
        macd = ta_indicators.get("macd")
        macd_signal = ta_indicators.get("macd_signal")
        stoch_k = ta_indicators.get("stoch_k")
        sma_5 = ta_indicators.get("sma_5")
        sma_20 = ta_indicators.get("sma_20")
        bb_pos = ta_indicators.get("bb_position")
        parts = []
        if rsi is not None:
            zone = "перекупленность" if rsi > 70 else ("перепроданность" if rsi < 30 else "нейтрально")
            parts.append(f"RSI={rsi:.0f} ({zone})")
        if macd is not None and macd_signal is not None:
            cross = "MACD>сигнал (бычий)" if macd > macd_signal else "MACD<сигнал (медвежий)"
            parts.append(f"MACD={macd:.2f}, {cross}")
        if stoch_k is not None:
            parts.append(f"Stochastic K={stoch_k:.0f}")
        if sma_5 is not None and sma_20 is not None:
            trend = "SMA5>SMA20 (растущий тренд)" if sma_5 > sma_20 else "SMA5<SMA20 (падающий тренд)"
            parts.append(trend)
        if bb_pos is not None:
            parts.append(f"BB pos={bb_pos:.2f}")
        if parts:
            ta_block = "Технический анализ:\n  " + "; ".join(parts) + "\n"

    user_message = (
        f"Акция: {ticker}, Цена: {current_price:.2f}\n"
        f"Общий сигнал: {trading_signal}\n"
        f"Средний R²: {r2_avg:.3f}, Средняя Direction Accuracy: {direction_accuracy_avg:.1f}%\n"
        f"Данные по моделям:\n{models_summary}\n"
        f"{ta_block}"
        f"\nAGREE или DISAGREE? Кратко обоснуй."
    )

    # Try Groq first, fall back to Ollama
    # Fallback chain: Groq → Mistral → Ollama
    raw = (
        _get_groq_response(user_message)
        or _get_mistral_response(user_message)
        or _get_ollama_response(user_message)
    )

    if raw is None:
        errs = get_last_errors()
        explanation_parts = []
        if errs.get("groq"):
            explanation_parts.append(f"Groq: {errs['groq']}")
        if errs.get("ollama"):
            explanation_parts.append(f"Ollama: {errs['ollama']}")
        explanation = "; ".join(explanation_parts) or "LLM недоступен"
        return {
            "answer": "UNAVAILABLE",
            "reasoning": "",
            "explanation": explanation,
        }

    # Парсим вердикт + причину.
    # Ожидаемый формат: первое слово AGREE/DISAGREE, дальше — текст.
    raw_upper = raw.upper()
    if "DISAGREE" in raw_upper.split("\n", 1)[0]:
        verdict = "DISAGREE"
    elif "AGREE" in raw_upper.split("\n", 1)[0]:
        verdict = "AGREE"
    else:
        return {
            "answer": "UNAVAILABLE",
            "reasoning": raw[:500],
            "explanation": f"Неожиданный формат ответа LLM",
        }

    # Извлекаем обоснование — всё, что после первой строки с вердиктом.
    reasoning = raw
    for marker in ("DISAGREE", "AGREE", "Disagree", "Agree", "disagree", "agree"):
        idx = reasoning.find(marker)
        if idx >= 0:
            reasoning = reasoning[idx + len(marker):]
            break
    reasoning = reasoning.lstrip(" .,:—-\n").strip()
    if not reasoning:
        reasoning = (
            "Сигнал подтверждён." if verdict == "AGREE"
            else "Сигнал не подтверждён."
        )
    # Ограничиваем длину для UI/БД
    reasoning = reasoning[:1000]

    # ============================================================
    # Пост-валидация: страховка от галлюцинаций LLM (особенно 8B моделей).
    # Считаем "рекомендованный" вердикт по тем же правилам что в промпте.
    # ============================================================
    rule_verdict = _rule_based_verdict(models_data)

    # Также детектим явную числовую ошибку в reasoning
    # (например "Direction 62.7% ниже порога 55%")
    reasoning_has_error = _reasoning_has_numeric_error(reasoning, models_data)

    if rule_verdict and (rule_verdict != verdict or reasoning_has_error):
        if rule_verdict != verdict:
            logger.warning(
                "LLM verdict (%s) противоречит правилам (%s). Использую правило.",
                verdict, rule_verdict,
            )
        if reasoning_has_error:
            logger.warning(
                "Reasoning от LLM содержит арифметическую ошибку — заменяю на правило-генерируемое."
            )
        verdict = rule_verdict
        # Полностью пересобираем reasoning по правилам — гарантированно точное
        reasoning = _build_rule_reasoning(models_data, rule_verdict)

    return {
        "answer": verdict,
        "reasoning": reasoning,
        "explanation": reasoning,
    }


def _reasoning_has_numeric_error(text: str, models_data: list) -> bool:
    """
    Детектит явную ошибку в reasoning вида:
      "Direction Accuracy составляет 62.7%, что ниже порога 55%"
    (62.7 НЕ ниже 55 — это галлюцинация).
    """
    if not text or not models_data:
        return False
    import re
    # Берём direction_accuracy из единственной модели в payload (per-row case)
    direction_vals = [m.get("direction_accuracy") for m in models_data
                      if m.get("direction_accuracy") is not None]
    if not direction_vals:
        return False
    da = float(direction_vals[0])

    # Ищем фразы "X% ниже" или "X% выше" где X — конкретное число
    lower_pattern = re.compile(
        r"(\d{1,2}(?:[.,]\d+)?)\s*%[^.]{0,40}ниже\s+порога?\s+(\d{1,2})",
        re.IGNORECASE,
    )
    higher_pattern = re.compile(
        r"(\d{1,2}(?:[.,]\d+)?)\s*%[^.]{0,40}выше\s+порога?\s+(\d{1,2})",
        re.IGNORECASE,
    )
    for m in lower_pattern.finditer(text):
        val = float(m.group(1).replace(",", "."))
        threshold = float(m.group(2))
        # Если LLM пишет "X% ниже порога Y", но фактически X >= Y → ошибка
        if abs(val - da) < 1.5 and val >= threshold:
            return True
    for m in higher_pattern.finditer(text):
        val = float(m.group(1).replace(",", "."))
        threshold = float(m.group(2))
        if abs(val - da) < 1.5 and val < threshold:
            return True
    return False


def _build_rule_reasoning(models_data: list, verdict: str) -> str:
    """Технически точное обоснование вердикта на основе численных правил."""
    if not models_data:
        return f"Оценка по правилам: {verdict}"

    m = models_data[0]  # per-row case
    parts = []
    da = m.get("direction_accuracy")
    pf = m.get("profit_factor")
    live = m.get("recent_hit_rate_30d")
    n = m.get("recent_samples_30d") or 0

    if live is not None and n >= 5:
        cmp = "выше" if live >= 55 else ("ниже" if live < 45 else "в серой зоне")
        parts.append(f"LIVE Direction за 30д = {live:.1f}% ({n} прогнозов), {cmp} ключевых порогов")
    elif da is not None:
        cmp = "выше" if da >= 55 else "ниже"
        parts.append(f"Direction Accuracy = {da:.1f}% ({cmp} порога 55%)")

    if pf is not None:
        if pf >= 1.0:
            parts.append(f"Profit Factor = {pf:.2f} (>=1.0, прибыльная)")
        else:
            parts.append(f"Profit Factor = {pf:.2f} (<1.0, в минусе)")

    body = ". ".join(parts)
    if verdict == "AGREE":
        return f"{body}. Метрики достаточно надёжны — доверять сигналу можно."
    else:
        return f"{body}. По правилам сигналу доверять не стоит."


def _rule_based_verdict(models_data: list) -> Optional[str]:
    """
    Алгоритмическая (без LLM) оценка по тем же правилам что в SYSTEM_PROMPT.
    Используется для пост-валидации LLM-ответа.
    Возвращает 'AGREE' / 'DISAGREE' или None если данных мало.

    Работает корректно как для per-row (1 модель в списке) так и для агрегатов.
    """
    if not models_data:
        return None

    # Усредняем по списку моделей (для per-row — список из 1)
    direction_vals = []
    live30_vals = []
    live30_samples = []
    pf_vals = []
    for m in models_data:
        d = m.get("direction_accuracy")
        if d is not None:
            direction_vals.append(float(d))
        live = m.get("recent_hit_rate_30d")
        if live is not None:
            live30_vals.append(float(live))
        ns = m.get("recent_samples_30d") or 0
        live30_samples.append(int(ns))
        pf = m.get("profit_factor")
        if pf is not None and float(pf) > 0:
            pf_vals.append(float(pf))

    avg_direction = sum(direction_vals) / len(direction_vals) if direction_vals else None
    avg_live = sum(live30_vals) / len(live30_vals) if live30_vals else None
    min_samples = min(live30_samples) if live30_samples else 0
    avg_pf = sum(pf_vals) / len(pf_vals) if pf_vals else None

    # Если LIVE достаточно — опираемся на него
    if avg_live is not None and min_samples >= 5:
        if avg_live >= 55:
            return "AGREE"
        if avg_live < 45:
            return "DISAGREE"
        # 45-55: смотрим historical direction
        if avg_direction is not None and avg_direction >= 55:
            return "AGREE"
        return "DISAGREE"

    # LIVE нет/мало — fallback на исторические
    if avg_direction is None:
        return None
    if avg_direction >= 55:
        if avg_pf is None or avg_pf >= 1.0:
            return "AGREE"
    if avg_direction < 50:
        return "DISAGREE"
    if avg_pf is not None and avg_pf < 0.8:
        return "DISAGREE"
    # Серая зона 50-55% Direction с нормальным PF — DISAGREE (осторожность)
    return "DISAGREE"


# ============================================================
# Развёрнутый отчёт по тикеру (использует все модели + TA + новости)
# ============================================================

REPORT_SYSTEM_PROMPT = """Ты — главный аналитик инвестиционного фонда с 20-летним стажем.

ЗАДАЧА: подготовь развёрнутый аналитический отчёт по конкретной акции на основе:
1. Прогнозов всех ML/DL моделей (название, сигнал, R², historical Direction,
   а также LIVE last30d / last5d — реальная точность последних прогнозов)
2. Технических индикаторов (RSI, MACD, Stochastic, тренд, BB)
3. Новостного фона (если передан)

ИЕРАРХИЯ ДОВЕРИЯ К МОДЕЛЯМ (важно!):
1. LIVE last30d — ГЛАВНЫЙ показатель. Реальная точность модели за
   последние 30 живых прогнозов. >55% — модель доказала адекватность,
   <45% — модель сейчас ошибается чаще монетки.
2. LIVE last5d — текущий импульс модели (если last30d ~50%).
3. hist_dir (historical Direction Accuracy) — % правильных направлений
   на тестовой выборке во время обучения. Хорошая исторически — > 55%.

⚠️ R² (коэффициент детерминации) — ИГНОРИРУЙ его при оценке.
Модели предсказывают доходность в %, поэтому R² для них почти всегда
БЛИЗОК К НУЛЮ или ОТРИЦАТЕЛЬНЫЙ — это нормально и не означает что
модель плохая. Возвраты на дневных свечах — это в основном шум, и
R² < 0 буквально значит "модель предсказывает движение хуже чем
если бы говорить 'средняя доходность = 0'". Direction Accuracy и
LIVE hit rate — единственные метрики, которые показывают
реальную полезность модели.

ВЗВЕШЕННЫЙ КОНСЕНСУС:
Если в данных есть "ВЗВЕШЕННЫЙ консенсус" — он считается с учётом
live-точности (модели с last30d >= 55% имеют вес 2x, с <45% — 0.5x).
Доверяй ему больше чем простому распределению.

МЕТРИКИ КАЧЕСТВА:
- LIVE last30d > 55% отлично, 50-55% средне, <50% слабо
- Direction > 60% отлично, 55-60% хорошо, <55% слабо
- R²: НЕ ОЦЕНИВАЙ. Это техническая метрика, для returns близка к 0.

ФОРМАТ ОТВЕТА (строго придерживайся!):
ВЕРДИКТ: <BUY или SELL или HOLD>
УВЕРЕННОСТЬ: <высокая, средняя или низкая>
ЦЕНА ВХОДА: <число>
ЦЕЛЕВАЯ ЦЕНА: <число>
СТОП-ЛОСС: <число>

КОНСЕНСУС МОДЕЛЕЙ:
<2-3 предложения: насколько модели согласованы, кто лидирует по R², какой сигнал>

ТЕХНИЧЕСКИЙ АНАЛИЗ:
<2-3 предложения: что говорят индикаторы, тренд, зоны>

НОВОСТНОЙ ФОН:
<1-2 предложения; если новостей нет — напиши "новостных данных нет">

РИСКИ:
<1-2 предложения: что может пойти не так>

РЕКОМЕНДАЦИЯ:
<1 предложение: конкретное действие со ссылкой на ценовые уровни>

ПРАВИЛА ДЛЯ ЦЕН:
- ЦЕНА ВХОДА — около текущей цены или чуть лучше (для BUY — равно или ниже; для SELL — равно или выше).
- ЦЕЛЕВАЯ ЦЕНА: для BUY — выше текущей на 2-7%; для SELL — ниже на 2-7%; для HOLD — равна текущей.
- СТОП-ЛОСС: для BUY — на 1.5-3% ниже цены входа; для SELL — на 1.5-3% выше; для HOLD — поставь 0.
- Указывай числа в той же валюте/масштабе, что и текущая цена. Без знаков $ и %, только число.

Пиши на русском. Без markdown, эмодзи, заголовков типа ##. Только текст и метки разделов как показано."""


def generate_ticker_report(
    ticker: str,
    current_price: float,
    models_data: list,
    ta_indicators: Optional[dict] = None,
    news_items: Optional[list] = None,
    data_date: Optional[str] = None,
    prediction_date: Optional[str] = None,
) -> dict:
    """
    Генерирует развёрнутый отчёт по тикеру через LLM.

    Args:
        ticker: SBER, OZON, ...
        current_price: текущая цена
        models_data: список dict с моделями
            [{'model_name', 'signal', 'r2', 'direction_accuracy'}, ...]
        ta_indicators: dict с RSI/MACD/Stoch/SMA/BB
        news_items: список dict [{'title', 'date', 'source'?}, ...]  # на будущее

    Returns:
        dict с полями:
          verdict (BUY/SELL/HOLD/UNKNOWN),
          confidence (высокая/средняя/низкая/—),
          sections {consensus, technical, news, risks, recommendation},
          report (полный текст),
          error (если что-то пошло не так)
    """
    # ----- Сборка пользовательского сообщения -----
    if not models_data:
        models_block = "Нет данных по моделям."
    else:
        lines = []
        signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0, "NEUTRAL": 0}

        # Если есть данные по live-точности — также собираем
        # ВЗВЕШЕННЫЙ консенсус: модели с recent_hit_rate >= 55% имеют вес 2x.
        weighted_signals = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "NEUTRAL": 0.0}

        for m in models_data:
            sig = (m.get("signal") or "?").upper()
            signals_count[sig] = signals_count.get(sig, 0) + 1
            r2 = m.get("r2") or 0
            da = m.get("direction_accuracy") or 0
            hit5 = m.get("recent_hit_rate_5d")
            hit30 = m.get("recent_hit_rate_30d")
            n5 = m.get("recent_samples_5d", 0)
            n30 = m.get("recent_samples_30d", 0)

            # Расчёт веса в консенсусе
            weight = 1.0
            if hit30 is not None and hit30 >= 55:
                weight = 2.0
            elif hit30 is not None and hit30 < 45:
                weight = 0.5
            weighted_signals[sig] = weighted_signals.get(sig, 0.0) + weight

            # Форматируем строку с приоритетом на live-точность
            live_part = ""
            if hit30 is not None and n30 >= 3:
                live_part = f", 🔥 LIVE last30d={hit30:.0f}% ({n30} прогн.)"
                if hit5 is not None and n5 >= 2:
                    live_part += f", last5d={hit5:.0f}%"
            else:
                live_part = ", LIVE: нет статистики"

            # Формируем строку: главное — Direction и LIVE, R² показываем мелко
            lines.append(
                f"  - {m.get('model_name','?')}: {sig}, "
                f"hist_dir={da:.1f}%{live_part} (R²={r2:.3f} — игнорируй)"
            )

        total = sum(signals_count.values())
        consensus_line = ", ".join(
            f"{k}={v}/{total}" for k, v in signals_count.items() if v > 0
        )

        # Взвешенный консенсус (если есть значимые веса)
        wtotal = sum(weighted_signals.values())
        if wtotal > 0:
            weighted_line = ", ".join(
                f"{k}={v:.1f}" for k, v in weighted_signals.items() if v > 0
            )
            weighted_block = f"\nВЗВЕШЕННЫЙ консенсус (по live-точности): {weighted_line}"
        else:
            weighted_block = ""

        models_block = (
            f"Распределение сигналов (равные веса): {consensus_line}"
            + weighted_block + "\n"
            + "\n".join(lines)
        )

    ta_block = "Технических индикаторов нет."
    if ta_indicators:
        parts = []
        rsi = ta_indicators.get("rsi")
        if rsi is not None:
            zone = (
                "перекупленность" if rsi > 70
                else "перепроданность" if rsi < 30 else "нейтрально"
            )
            parts.append(f"RSI={rsi:.0f} ({zone})")
        macd = ta_indicators.get("macd")
        macd_sig = ta_indicators.get("macd_signal")
        if macd is not None and macd_sig is not None:
            cross = "MACD выше сигнальной (бычий)" if macd > macd_sig \
                else "MACD ниже сигнальной (медвежий)"
            parts.append(f"MACD={macd:.2f}; {cross}")
        stoch = ta_indicators.get("stoch_k")
        if stoch is not None:
            parts.append(f"Stochastic K={stoch:.0f}")
        sma5 = ta_indicators.get("sma_5")
        sma20 = ta_indicators.get("sma_20")
        if sma5 is not None and sma20 is not None:
            parts.append(
                "SMA5>SMA20 (растущий тренд)" if sma5 > sma20
                else "SMA5<SMA20 (падающий тренд)"
            )
        bb = ta_indicators.get("bb_position")
        if bb is not None:
            parts.append(f"BB позиция={bb:.2f}")
        # Дневной диапазон (high/low) последней свечи
        last_high = ta_indicators.get("last_high")
        last_low = ta_indicators.get("last_low")
        last_price = ta_indicators.get("last_price")
        if last_high is not None and last_low is not None:
            spread = (last_high - last_low)
            spread_pct = (spread / last_price * 100) if last_price else 0
            parts.append(
                f"Дневной диапазон вчера: high={last_high:.2f}, low={last_low:.2f} "
                f"(размах {spread_pct:.2f}%)"
            )
        if parts:
            ta_block = "; ".join(parts)

    if news_items:
        news_lines = [
            f"  - {n.get('date','?')}: {n.get('title','')[:140]}"
            for n in news_items[:5]
        ]
        news_block = "\n".join(news_lines)
    else:
        news_block = "новостных данных нет"

    dates_block = ""
    if data_date or prediction_date:
        dates_block = (
            f"Данные актуальны на: {data_date or '?'}\n"
            f"Прогноз делается на: {prediction_date or '?'} (следующий торговый день)\n\n"
        )

    user_message = (
        f"Тикер: {ticker}\n"
        f"Текущая цена (close на {data_date or '?'}): {current_price:.2f}\n"
        f"{dates_block}"
        f"=== ПРОГНОЗЫ МОДЕЛЕЙ ===\n{models_block}\n\n"
        f"=== ТЕХНИЧЕСКИЙ АНАЛИЗ (на {data_date or '?'}) ===\n{ta_block}\n\n"
        f"=== НОВОСТНОЙ ФОН ===\n{news_block}\n\n"
        f"Подготовь отчёт строго в указанном формате."
    )

    # ----- Запрос к LLM -----
    raw = (
        _get_groq_response_custom(user_message, REPORT_SYSTEM_PROMPT, max_tokens=600)
        or _get_mistral_response(user_message, REPORT_SYSTEM_PROMPT, max_tokens=600)
        or _get_ollama_response_custom(user_message, REPORT_SYSTEM_PROMPT)
    )

    if not raw:
        errs = get_last_errors()
        return {
            "verdict": "UNKNOWN",
            "confidence": "—",
            "sections": {},
            "report": "",
            "error": "; ".join(f"{k}: {v}" for k, v in errs.items() if v) or "LLM недоступен",
        }

    # ----- Парсинг разделов -----
    sections = _parse_report_sections(raw)
    verdict = sections.pop("_verdict", "UNKNOWN")
    confidence = sections.pop("_confidence", "—")
    entry_price = sections.pop("_entry_price", None)
    target_price = sections.pop("_target_price", None)
    stop_loss = sections.pop("_stop_loss", None)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "sections": sections,
        "report": raw,
        "error": None,
    }


def _parse_report_sections(text: str) -> dict:
    """Разбивает текст отчёта на разделы по меткам."""
    import re

    result = {
        "_verdict": "UNKNOWN",
        "_confidence": "—",
        "_entry_price": None,
        "_target_price": None,
        "_stop_loss": None,
        "consensus": "",
        "technical": "",
        "news": "",
        "risks": "",
        "recommendation": "",
    }

    # Вердикт
    m = re.search(r"ВЕРДИКТ[:\s]+(BUY|SELL|HOLD|NEUTRAL)", text, re.IGNORECASE)
    if m:
        result["_verdict"] = m.group(1).upper()

    # Уверенность
    m = re.search(r"УВЕРЕННОСТЬ[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        result["_confidence"] = m.group(1).lower()

    # Цены — извлекаем числа после меток
    def _parse_price(label: str):
        pat = label + r"[:\s]+([-+]?\d+(?:[.,]\d+)?)"
        mp = re.search(pat, text, re.IGNORECASE)
        if mp:
            try:
                return float(mp.group(1).replace(",", "."))
            except ValueError:
                return None
        return None

    result["_entry_price"] = _parse_price(r"ЦЕНА\s+ВХОДА")
    result["_target_price"] = _parse_price(r"ЦЕЛЕВ\w*\s+ЦЕНА")
    result["_stop_loss"] = _parse_price(r"СТОП[-\s]?ЛОСС")

    # Разделы — ищем по ключевым словам
    patterns = {
        "consensus": r"КОНСЕНСУС МОДЕЛЕЙ[:\s]*\n?(.+?)(?=\n[A-ЯЁ]{3,}[ A-ЯЁ]*:|\Z)",
        "technical": r"ТЕХНИЧЕСКИЙ АНАЛИЗ[:\s]*\n?(.+?)(?=\n[A-ЯЁ]{3,}[ A-ЯЁ]*:|\Z)",
        "news": r"НОВОСТНОЙ ФОН[:\s]*\n?(.+?)(?=\n[A-ЯЁ]{3,}[ A-ЯЁ]*:|\Z)",
        "risks": r"РИСКИ[:\s]*\n?(.+?)(?=\n[A-ЯЁ]{3,}[ A-ЯЁ]*:|\Z)",
        "recommendation": r"РЕКОМЕНДАЦИЯ[:\s]*\n?(.+?)\Z",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            result[key] = m.group(1).strip()

    return result


TA_SUMMARY_SYSTEM_PROMPT = """Ты — опытный технический аналитик. Твоя задача — дать ОДНОЗНАЧНЫЙ
короткий вывод по техническим индикаторам акции.

⚠️ ВАЖНО — БАЗОВЫЕ ПРАВИЛА ИНТЕРПРЕТАЦИИ (не путать!):

RSI (индекс относительной силы):
- RSI > 70 = ПЕРЕКУПЛЕННОСТЬ = МЕДВЕЖИЙ сигнал (цена слишком высоко,
  вероятна коррекция ВНИЗ)
- RSI < 30 = ПЕРЕПРОДАННОСТЬ = БЫЧИЙ сигнал (цена слишком низко,
  вероятен отскок ВВЕРХ)
- RSI 30-70 = нейтральная зона

Stochastic K — те же правила что RSI:
- K > 80 = перекупленность = МЕДВЕЖИЙ
- K < 20 = перепроданность = БЫЧИЙ

Bollinger Bands (BB позиция от 0 до 1):
- BB > 0.8 = цена у верхней границы = перекупленность = МЕДВЕЖИЙ
- BB < 0.2 = цена у нижней границы = перепроданность = БЫЧИЙ
- BB около 0.5 = нейтрально

MACD (тренд/импульс):
- MACD > сигнальной линии = БЫЧИЙ импульс
- MACD < сигнальной линии = МЕДВЕЖИЙ импульс

SMA (тренд):
- SMA5 > SMA20 = восходящий тренд = БЫЧИЙ
- SMA5 < SMA20 = нисходящий тренд = МЕДВЕЖИЙ

ФОРМАТ ОТВЕТА (строго!):
ВЫВОД: <БЫЧИЙ или МЕДВЕЖИЙ или СМЕШАННЫЙ или НЕЙТРАЛЬНЫЙ>
СИЛА: <сильный или умеренный или слабый>
КРАТКО: <одно предложение по-русски, ИСПОЛЬЗУЙ КОНКРЕТНЫЕ ЦИФРЫ
        и НЕ ПРОТИВОРЕЧЬ вердикту>
КЛЮЧЕВЫЕ ИНДИКАТОРЫ: <2-3 индикатора через запятую>

ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:
- RSI=22, MACD растёт → "БЫЧИЙ. Перепроданность по RSI (22) даёт
  потенциал отскока вверх, MACD подтверждает разворот."
- RSI=85, BB=0.95 → "МЕДВЕЖИЙ. Сильная перекупленность (RSI 85, BB 0.95)
  указывает на высокую вероятность коррекции вниз."

⚠️ САМОПРОВЕРКА:
Если упоминаешь "перекупленность" → вердикт ДОЛЖЕН быть МЕДВЕЖИЙ или СМЕШАННЫЙ.
Если упоминаешь "перепроданность" → вердикт ДОЛЖЕН быть БЫЧИЙ или СМЕШАННЫЙ.
НЕЛЬЗЯ говорить "перекупленность даёт потенциал роста" — это ошибка.

Без markdown, без эмодзи. Точно следуй формату."""


def summarize_ta_indicators(ticker: str, ta_indicators: dict) -> dict:
    """
    Короткий вывод по техническим индикаторам.
    Возвращает {verdict, strength, summary, key_indicators, error}.
    """
    if not ta_indicators:
        return {"verdict": "UNKNOWN", "strength": "—",
                "summary": "", "key_indicators": "",
                "error": "Нет данных TA"}

    # Собираем компактный текст с индикаторами
    parts = []
    rsi = ta_indicators.get("rsi")
    if rsi is not None:
        parts.append(f"RSI(14)={rsi:.1f}")
    macd = ta_indicators.get("macd")
    macd_sig = ta_indicators.get("macd_signal")
    if macd is not None and macd_sig is not None:
        parts.append(f"MACD={macd:.2f}, сигн={macd_sig:.2f}")
    stoch = ta_indicators.get("stoch_k")
    if stoch is not None:
        parts.append(f"Stoch K={stoch:.1f}")
    sma5 = ta_indicators.get("sma_5")
    sma20 = ta_indicators.get("sma_20")
    if sma5 is not None and sma20 is not None:
        parts.append(f"SMA5={sma5:.2f}, SMA20={sma20:.2f}")
    bb = ta_indicators.get("bb_position")
    if bb is not None:
        parts.append(f"BB позиция={bb:.2f}")
    price = ta_indicators.get("last_price")
    if price is not None:
        parts.append(f"Цена={price:.2f}")

    if not parts:
        return {"verdict": "UNKNOWN", "strength": "—",
                "summary": "", "key_indicators": "",
                "error": "Нет валидных индикаторов"}

    user_message = f"Акция: {ticker}\nПоказатели: {'; '.join(parts)}\n\nДай вывод в требуемом формате."

    raw = (
        _get_groq_response_custom(user_message, TA_SUMMARY_SYSTEM_PROMPT, max_tokens=200)
        or _get_mistral_response(user_message, TA_SUMMARY_SYSTEM_PROMPT, max_tokens=200)
        or _get_ollama_response_custom(user_message, TA_SUMMARY_SYSTEM_PROMPT)
    )
    if not raw:
        errs = get_last_errors()
        return {
            "verdict": "UNKNOWN", "strength": "—",
            "summary": "", "key_indicators": "",
            "error": "; ".join(f"{k}: {v}" for k, v in errs.items() if v) or "LLM недоступен",
        }

    # Парсим разделы
    import re as _re
    result = {"verdict": "UNKNOWN", "strength": "—",
              "summary": "", "key_indicators": "", "error": None}

    m = _re.search(r"ВЫВОД[:\s]+(\S+)", raw, _re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).strip().upper().rstrip('.,;:')

    m = _re.search(r"СИЛА[:\s]+(\S+)", raw, _re.IGNORECASE)
    if m:
        result["strength"] = m.group(1).strip().lower().rstrip('.,;:')

    m = _re.search(r"КРАТКО[:\s]+(.+?)(?:\n[A-ЯЁ]{3,}|\Z)", raw, _re.DOTALL | _re.IGNORECASE)
    if m:
        result["summary"] = m.group(1).strip()

    m = _re.search(r"КЛЮЧЕВЫЕ ИНДИКАТОРЫ[:\s]+(.+?)(?:\n[A-ЯЁ]{3,}|\Z)", raw, _re.DOTALL | _re.IGNORECASE)
    if m:
        result["key_indicators"] = m.group(1).strip()

    return result


TA_EXPLAIN_SYSTEM_PROMPT = """Ты — опытный технический аналитик. Объясняй простым языком новичкам.

ЗАДАЧА: получив значения технических индикаторов по акции, дай краткое (4-6 предложений)
объяснение что они показывают в данный момент. Без рекомендаций "покупать/продавать" —
только интерпретация состояния.

Структура ответа:
1. Тренд (по SMA): краткосрочный / долгосрочный.
2. Импульс (по RSI, MACD, Stochastic): набирает / теряет / в равновесии.
3. Зоны (перекупленность / перепроданность / нейтрально).
4. Волатильность (по диапазону вчера, BB позиции).

Пиши без markdown, без эмодзи, без заголовков типа ##. Только текст с упоминанием цифр.
Один связный абзац."""


def explain_ta_indicators(ticker: str, ta_indicators: dict) -> dict:
    """
    Просит LLM объяснить значения технических индикаторов простыми словами.
    Возвращает {explanation, error}.
    """
    if not ta_indicators:
        return {"explanation": "", "error": "Нет данных технического анализа"}

    # Сборка запроса
    parts = []
    rsi = ta_indicators.get("rsi")
    if rsi is not None:
        parts.append(f"RSI(14) = {rsi:.1f}")
    macd = ta_indicators.get("macd")
    macd_sig = ta_indicators.get("macd_signal")
    if macd is not None:
        if macd_sig is not None:
            parts.append(f"MACD = {macd:.2f}, сигнальная = {macd_sig:.2f}")
        else:
            parts.append(f"MACD = {macd:.2f}")
    stoch = ta_indicators.get("stoch_k")
    if stoch is not None:
        parts.append(f"Stochastic K = {stoch:.1f}")
    sma5 = ta_indicators.get("sma_5")
    sma20 = ta_indicators.get("sma_20")
    last_price = ta_indicators.get("last_price")
    if sma5 is not None and sma20 is not None:
        parts.append(f"SMA(5) = {sma5:.2f}, SMA(20) = {sma20:.2f}")
    if last_price is not None:
        parts.append(f"Текущая цена = {last_price:.2f}")
    bb = ta_indicators.get("bb_position")
    if bb is not None:
        parts.append(f"Положение в полосах Боллинджера = {bb:.2f} (0 = нижняя, 1 = верхняя)")
    high = ta_indicators.get("last_high")
    low = ta_indicators.get("last_low")
    if high is not None and low is not None:
        rng = high - low
        rng_pct = (rng / last_price * 100) if last_price else 0
        parts.append(
            f"Дневной диапазон: high = {high:.2f}, low = {low:.2f} "
            f"(размах {rng_pct:.2f}%)"
        )

    if not parts:
        return {"explanation": "", "error": "Нет валидных индикаторов для интерпретации"}

    user_message = (
        f"Акция: {ticker}\n\nПоказатели:\n  - " + "\n  - ".join(parts) +
        "\n\nОбъясни что они показывают в данный момент."
    )

    raw = (
        _get_groq_response_custom(user_message, TA_EXPLAIN_SYSTEM_PROMPT, max_tokens=350)
        or _get_mistral_response(user_message, TA_EXPLAIN_SYSTEM_PROMPT, max_tokens=350)
        or _get_ollama_response_custom(user_message, TA_EXPLAIN_SYSTEM_PROMPT)
    )

    if not raw:
        errs = get_last_errors()
        return {
            "explanation": "",
            "error": "; ".join(f"{k}: {v}" for k, v in errs.items() if v) or "LLM недоступен",
        }

    return {"explanation": raw.strip(), "error": None}


def _get_groq_response_custom(user_message: str, system_prompt: str,
                              max_tokens: int = 500) -> Optional[str]:
    """Версия Groq-запроса с кастомным системным промптом и лимитом токенов."""
    global _LAST_GROQ_ERROR
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        _LAST_GROQ_ERROR = "GROQ_API_KEY не задан"
        return None

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip().strip('"').strip("'")

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        _LAST_GROQ_ERROR = None
        return completion.choices[0].message.content.strip()
    except Exception as e:
        _LAST_GROQ_ERROR = f"{type(e).__name__}: {e}"
        logger.error("Groq API error (model=%s): %s", model, e)
        return None


def _get_ollama_response_custom(user_message: str, system_prompt: str) -> Optional[str]:
    """Аналогично для Ollama."""
    global _LAST_OLLAMA_ERROR
    url = os.getenv("LLM_URL", "http://localhost:11434/api/chat")
    model = os.getenv("LLM_MODEL", "llama3")
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            _LAST_OLLAMA_ERROR = None
            return resp.json()["message"]["content"].strip()
        _LAST_OLLAMA_ERROR = f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        _LAST_OLLAMA_ERROR = f"{type(e).__name__}: {e}"
        logger.error("Ollama error: %s", e)
    return None
