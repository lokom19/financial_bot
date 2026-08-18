"""
Новостной фид по российским акциям.

Источники (в порядке приоритета):
  1. SearXNG (self-hosted metasearch) — актуальные новости из Google/Bing/Yandex
  2. smart-lab.ru (HTML-парсинг) — фолбэк если SearXNG недоступен

Кеш в памяти процесса на 30 минут (per-ticker).
"""
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

# Поисковые запросы для SearXNG (более точные, чем для smart-lab)
TICKER_TO_SEARXNG_QUERY = {
    "SBER":  "Сбербанк акции новости",
    "OZON":  "Ozon акции новости",
    "VTBR":  "ВТБ банк акции новости",
    "GAZP":  "Газпром акции новости",
    "LKOH":  "Лукойл акции новости",
    "ROSN":  "Роснефть акции новости",
    "YDEX":  "Яндекс акции новости",
    "MTSS":  "МТС акции новости",
    "AFLT":  "Аэрофлот акции новости",
    "TCSG":  "Т-Банк ТКС акции новости",
    "HEAD":  "HeadHunter акции новости",
}

# Ключи поиска для smart-lab (фолбэк)
TICKER_TO_SEARCH = {
    "SBER":  "сбер",
    "OZON":  "ozon",
    "VTBR":  "втб",
    "GAZP":  "газпром",
    "LKOH":  "лукойл",
    "ROSN":  "роснефть",
    "TCSG":  "тинькофф",
    "YNDX":  "яндекс",
    "YDEX":  "яндекс",
    "MTSS":  "мтс",
    "AFLT":  "аэрофлот",
    "HEAD":  "headhunter",
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


def _fetch_searxng(ticker: str, max_items: int) -> List[dict]:
    """
    Запрашивает SearXNG JSON API.
    Возвращает список новостей или [] при недоступности.
    """
    query = TICKER_TO_SEARXNG_QUERY.get(ticker.upper(), f"{ticker} акции новости")
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "news",
                "language": "ru-RU",
                "time_range": "week",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("SearXNG вернул %s для %s", resp.status_code, ticker)
            return []
        data = resp.json()
        results = data.get("results") or []
        news = []
        for r in results:
            title = (r.get("title") or "").strip()
            url = r.get("url") or ""
            if not title or not url:
                continue
            pub_date = None
            raw_date = r.get("publishedDate") or ""
            if raw_date:
                try:
                    pub_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date().isoformat()
                except Exception:
                    pass
            news.append({
                "date": pub_date,
                "date_raw": raw_date[:30],
                "title": title[:200],
                "url": url,
                "source": r.get("engine", "searxng"),
            })
            if len(news) >= max_items:
                break
        return news
    except Exception as e:
        logger.warning("SearXNG недоступен для %s: %s", ticker, e)
        return []


def _fetch_smartlab(ticker: str, max_items: int) -> List[dict]:
    """HTML-парсинг smart-lab.ru (фолбэк)."""
    search = TICKER_TO_SEARCH.get(ticker.upper(), ticker.lower())
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_news = []
    for page in range(1, 3):
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
            logger.warning("smart-lab fetch error for %s: %s", ticker, e)
            break
    return all_news


def fetch_news_for_ticker(ticker: str, max_items: int = 5,
                          max_pages: int = 2) -> List[dict]:
    """
    Возвращает [{date, title, url, source}].
    Пробует SearXNG первым; если пусто — smart-lab.
    Кеш 30 минут.
    """
    cache_key = (ticker.upper(), max_items)
    now = datetime.utcnow()

    cached = _CACHE.get(cache_key)
    if cached:
        ts, items = cached
        if (now - ts).total_seconds() < _CACHE_TTL_SECONDS:
            return items

    news = _fetch_searxng(ticker, max_items)

    if not news:
        logger.info("SearXNG: нет результатов для %s, пробую smart-lab", ticker)
        news = _fetch_smartlab(ticker, max_items)

    if news:
        _CACHE[cache_key] = (now, news)
    return news
