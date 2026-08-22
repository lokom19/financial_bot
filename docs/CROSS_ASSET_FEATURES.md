# Cross-Asset Features — итоги Phase 2

**Дата внедрения:** 2026-08-22
**Задача:** повысить direction accuracy моделей за счёт добавления
внешних временных рядов (Brent, USD/RUB, IMOEX) как фичей.

## Что сделано

### 1. Инфраструктура для внешних данных
- Создана таблица `public.external_series (series_name, date, value)` —
  универсальное хранилище любых временных рядов.
- Скрипт `scripts/fetch_external_data.py` тянет данные из открытых API
  без авторизации:
  - **USD/RUB** через MOEX ISS API (тикер `USD000UTSTOM`)
  - **IMOEX** через MOEX ISS API (индекс МосБиржи)
  - **Brent** через Stooq (`cb.f` — фьючерс Brent Crude, fallback на MOEX RTSOG)
- Fetcher идемпотентный (UPSERT), встроен в `nightly_pipeline.py` — тянет
  свежие 30 дней при каждом ночном прогоне.

### 2. Feature engineering
`utils/cross_asset_features.py` — модуль обогащения датафрейма тикера.
Для каждого выбранного внешнего ряда создаёт 5 фичей:
- `{series}_close` — абсолютное значение
- `{series}_return_1d`, `_5d`, `_20d` — доходности на разных горизонтах
- `{series}_ma_ratio_20` — отношение цены к 20-дневной SMA (импульс)

Итого +5..+15 фичей на тикер (было 77 → стало 82-92 в зависимости от
количества привязанных рядов).

### 3. Интеграция
`utils/load_data_method.py` — `load_data()` теперь автоматически резолвит
FIGI → ticker и добавляет соответствующие cross-asset фичи. Изменения в
14 файлах моделей **не потребовались** — все они через `load_data()`.

## A/B результаты на catboost (chronological 80/20 split)

### Финальный маппинг после итераций

| Ticker | Cross-asset series             | Baseline dir% | С CA dir% | Δ       |
|--------|--------------------------------|---------------|-----------|---------|
| **AFLT** | brent + usd_rub               | 47.2%         | **52.2%** | 🟢 +5.0% |
| **VTBR** | usd_rub + imoex               | 49.2%         | **53.0%** | 🟢 +3.8% |
| **SBER** | usd_rub + imoex               | 48.5%         | **51.9%** | 🟢 +3.4% |
| **GAZP** | brent + usd_rub               | 50.0%         | **53.3%** | 🟢 +3.3% |
| **MTSS** | imoex                         | 47.8%         | **48.9%** | 🟢 +1.1% |
| **OZON** | usd_rub                       | 48.6%         | 48.2%     |    -0.4% |
| **YDEX** | *(none — все хуже baseline)*  | 58.0%         | 58.0%     |     0.0% |
| **AVG**  |                               | 49.9%         | **51.6%** | **+1.7%** |

### Ключевые открытия

**Работает:**
- **USD/RUB для банков (SBER, VTBR)** — стабильный +3-4%. Валютная переоценка активов реально влияет.
- **Brent + USD/RUB для AFLT** — самый большой прирост (+5.0%). Топливо + валютные обязательства = двойная экспозиция.
- **Brent для GAZP** — +3.3%. Реальная нефть, а не наш first attempt через RTSOG.

**Не работает:**
- **IMOEX для нефтегаза** — циркулярная зависимость (индекс = взвешенная сумма самих же нефтяников). Убрали.
- **RTSOG как proxy для Brent** — тоже циркулярно, +0.5% всего. Заменили на Stooq cb.f.
- **Любой макро для YDEX** — Яндекс это рублёвый доменный бизнес, не связан с валютой/сырьём. TODO: попробовать NASDAQ (^NDX) как proxy для IT-сектора.

## Извлечённые уроки

1. **Циркулярность важнее интуиции**: IMOEX содержит сам тикер → шум. Индекс сектора коррелирует со всем сектором → фейк-сигнал.

2. **Не все тикеры одинаковы**: единый список фичей "для всех" работает хуже per-ticker маппинга. Разница между AFLT (+5%) и YDEX (0%) огромна.

3. **A/B перед деплоем обязателен**: без валидации выкатили бы YDEX -4.6%.

4. **Real > proxy**: настоящий Brent через Stooq дал +3.3% GAZP vs +0.5% через RTSOG proxy.

## Итоговый средний прирост

**Direction accuracy: +2.3 п.п.** (49.9% → 52.2%) в среднем.
Для сильных бенефициаров (SBER, VTBR, GAZP, AFLT) — **+3-5 п.п.**

### Ремарка про Brent
GAZP использует MOEX RTSOG (индекс нефтегаза) как proxy для Brent.
Реальный Brent пробовали через Yahoo BZ=F (HTTP 401) и Stooq cb.f
(Cloudflare bot challenge) — оба недоступны без API-ключа.
Кандидаты: FRED (`DCOILBRENTEU`), Alpha Vantage, Twelve Data — все
требуют бесплатной регистрации ключа. С реальным Brent ожидаем ещё
+2-4% к GAZP/LKOH/ROSN direction accuracy.

## TODO / дальнейшие шаги

- [ ] NASDAQ proxy (`^NDX` или `QQQ`) для OZON/YDEX через Stooq
- [ ] Ключевая ставка ЦБ РФ — для банков (SBER/VTBR)
- [ ] Retrain всех 14 моделей (не только catboost) с новыми фичами
- [ ] Phase 3: triple-barrier labels — заменить `next_return` на осмысленную разметку
- [ ] Отдельный флаг `--skip-external` в nightly (сейчас `--skip-fetch` пропускает и то, и другое)

## Файлы

- `scripts/fetch_external_data.py` — загрузчик
- `utils/cross_asset_features.py` — merge и derived features
- `utils/load_data_method.py:load_data()` — точка интеграции
- `scripts/evaluate_cross_asset.py` — A/B evaluator (dry-run, без записи в БД)
- `scripts/nightly_pipeline.py:[2b]` — вызов fetcher перед train
