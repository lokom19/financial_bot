"""
Новостной фид по российским акциям.

Источник: smart-lab.ru (HTML-парсинг блока news).
Для каждого тикера маппим на ключевые слова поиска.

Кеш в памяти процесса на 30 минут (per-ticker).
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Ключи поиска по тикерам (smart-lab ищет по русским формам)
TICKER_TO_SEARCH = {
    "SBER":  "сбер",
    "OZON":  "ozon",
    "VTBR":  "втб",
    "GAZP":  "газпром",
    "LKOH":  "лукойл",
    "ROSN":  "роснефть",
    "TCSG":  "тинькофф",
    "YNDX":  "яндекс",
    "MAGN":  "магнит",
    "MGNT":  "магнит",
}

MONTH_MAP = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
}

# In-memory кеш: {(ticker, max_items): (timestamp_loaded, [news])}
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 30 * 60


def _parse_smartlab_date(date_str: str) -> Optional[datetime]:
    """'27 июня 2026, 15:20' → datetime."""
    if not date_str:
        return None
    try:
        parts = date_str.replace(',', '').split()
        if len(parts) >= 3:
            day = int(parts[0])
            month = MONTH_MAP.get(parts[1])
            year = int(parts[2])
            if month:
                return datetime(year, month, day)
    except Exception:
        pass
    return None


def fetch_news_for_ticker(ticker: str, max_items: int = 5,
                          max_pages: int = 2) -> List[dict]:
    """
    Возвращает [{date: '2026-06-27', title: '...', url: '...', source: 'smart-lab'}].
    Кеш 30 минут.
    """
    cache_key = (ticker.upper(), max_items)
    now = datetime.utcnow()

    cached = _CACHE.get(cache_key)
    if cached:
        ts, items = cached
        if (now - ts).total_seconds() < _CACHE_TTL_SECONDS:
            return items

    search = TICKER_TO_SEARCH.get(ticker.upper(), ticker.lower())
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_news = []

    for page in range(1, max_pages + 1):
        url = (
            f"https://smart-lab.ru/search/topics/?blog=news&q={search}"
            if page == 1
            else f"https://smart-lab.ru/search/topics/page{page}/?q={search}&blog=news"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, 'lxml')
            topics = soup.find_all('div', class_=lambda x: x and 'topic' in (x or ''))
            if not topics:
                break

            for t in topics:
                title_elem = t.find('h2', class_='title')
                if not title_elem:
                    continue
                link = title_elem.find('a')
                if not link:
                    continue
                title = (link.get('title') or link.get_text(strip=True) or '').strip()
                href = link.get('href') or ''
                if href and not href.startswith('http'):
                    href = 'https://smart-lab.ru' + href

                date_elem = t.find('li', class_='date')
                date_raw = date_elem.get_text(strip=True)[:30] if date_elem else ''
                pub = _parse_smartlab_date(date_raw)

                all_news.append({
                    'date': pub.date().isoformat() if pub else None,
                    'date_raw': date_raw,
                    'title': title[:200],
                    'url': href,
                    'source': 'smart-lab',
                })
                if len(all_news) >= max_items:
                    break
            if len(all_news) >= max_items:
                break
        except Exception as e:
            logger.warning("news fetch error for %s: %s", ticker, e)
            break

    # Не кешируем пустые результаты — чтобы временные ошибки парсинга/сети
    # не блокировали повторные попытки на 30 минут.
    if all_news:
        _CACHE[cache_key] = (now, all_news)
    return all_news
