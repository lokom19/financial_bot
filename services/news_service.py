"""
Новостной фид по российским акциям.

Источники (в порядке приоритета):
  1. SearXNG (self-hosted metasearch) — актуальные новости из Brave/Yandex
  2. smart-lab.ru (HTML-парсинг) — фолбэк если SearXNG недоступен

Кеш в памяти процесса на 30 минут (per-ticker).
"""
import logging
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

TICKER_TO_SEARXNG_QUERY = {
    "SBER":  "Сбербанк акции новости",
    "OZON":  "Озон OZON акции новости",
    "VTBR":  "ВТБ банк акции новости",
    "GAZP":  "Газпром акции новости",
    "LKOH":  "Лукойл акции новости",
    "ROSN":  "Роснефть акции новости",
    "YDEX":  "Яндекс акции новости",
    "MTSS":  "МТС акции новости",
    "AFLT":  "Аэрофлот акции новости",
    "TCSG":  "Т-Банк ТКС акции новости",
    "HEAD":  "HeadHunter Хедхантер акции новости",
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

# Домены-агрегаторы котировок — не новости, убираем из результатов целиком
_SKIP_DOMAINS = {
    "investing.com", "ru.investing.com",
    "tradingview.com", "ru.tradingview.com",
    "moex.com", "finance.yahoo.com",
    "t-bank.ru", "tbank.ru", "tinkoff.ru",
    "bcs-express.ru",
}

# URL-паттерны страниц котировок — не статьи, пропускаем
_SKIP_URL_RE = re.compile(
    r'/quote/|/котировки|/ticker(?:/|$)|/stocks?(?:/|$)'
    r'|/chart(?:/|$)|/price(?:/|$)|/акции/[A-Z]{3,5}/?$',
    re.I,
)

# Домены с пейволлом или блокировкой ботов — держим SearXNG-сниппет, URL не качаем
_NO_FETCH_DOMAINS = {
    "vedomosti.ru", "bloomberg.com", "reuters.com",
    "wsj.com", "ft.com", "kommersant.ru",
}

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

MONTH_MAP = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
}

# In-memory кеш: {(ticker, max_items): (timestamp_loaded, [news])}
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 30 * 60


def _get_host(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _in_domains(url: str, domain_set: set) -> bool:
    host = _get_host(url)
    return any(host == d or host.endswith("." + d) for d in domain_set)


def _fetch_url_content(url: str, max_chars: int = 5000) -> str:
    """
    Загружает страницу статьи и извлекает основной текст.
    Возвращает первые max_chars символов или '' при ошибке/блокировке.
    """
    if _in_domains(url, _SKIP_DOMAINS | _NO_FETCH_DOMAINS):
        return ""
    try:
        resp = requests.get(url, headers=_FETCH_HEADERS, timeout=7, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript", "figure"]):
            tag.decompose()

        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"article|content|body|text|post", re.I))
            or soup.body
        )
        if not container:
            return ""

        parts, total = [], 0
        for p in container.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) < 40:
                continue
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break

        return " ".join(parts)[:max_chars].strip()
    except Exception as e:
        logger.debug("fetch_url_content failed for %s: %s", url, e)
        return ""


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
    Запрашивает max_items*3 результатов, фильтрует мусорные домены,
    возвращает до max_items записей или [] при недоступности.
    """
    query = TICKER_TO_SEARXNG_QUERY.get(ticker.upper(), f"{ticker} акции новости")
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
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
            if _in_domains(url, _SKIP_DOMAINS):
                logger.debug("skip domain: %s", _get_host(url))
                continue
            if _SKIP_URL_RE.search(url):
                logger.debug("skip url pattern: %s", url[:80])
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
                "snippet": (r.get("content") or "")[:400],
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
                    'snippet': '',
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


def _enrich_with_content(news: List[dict]) -> List[dict]:
    """
    Для каждой новости пробует загрузить страницу и заменить snippet
    на реальный текст статьи.
    """
    for item in news:
        url = item.get("url", "")
        if not url:
            continue
        existing = (item.get("snippet") or "").strip()
        fetched = _fetch_url_content(url)
        if fetched and len(fetched) > len(existing):
            item["snippet"] = fetched
            logger.debug("enriched %d chars from %s", len(fetched), url[:60])
    return news


def fetch_news_for_ticker(ticker: str, max_items: int = 5,
                          max_pages: int = 2,
                          enrich_content: bool = True) -> List[dict]:
    """
    Возвращает [{date, title, url, snippet, source}].
    Всегда берёт из обоих источников: SearXNG + smart-lab.
    Дедуплицирует по URL, обогащает текстом статьи.
    Кеш 30 минут.
    """
    cache_key = (ticker.upper(), max_items)
    now = datetime.utcnow()

    cached = _CACHE.get(cache_key)
    if cached:
        ts, items = cached
        if (now - ts).total_seconds() < _CACHE_TTL_SECONDS:
            return items

    searxng_news = _fetch_searxng(ticker, max_items)
    smartlab_news = _fetch_smartlab(ticker, max_items=3)

    # объединяем, дедуплицируем по URL
    seen_urls = set()
    combined = []
    for item in searxng_news + smartlab_news:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            combined.append(item)

    if not combined:
        logger.warning("Нет новостей для %s ни из одного источника", ticker)

    if combined and enrich_content:
        combined = _enrich_with_content(combined)

    if combined:
        _CACHE[cache_key] = (now, combined)
    return combined
