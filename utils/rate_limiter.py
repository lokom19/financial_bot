import time
import logging
import asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, rate_limit=30, per_seconds=60):
        self.rate_limit = rate_limit  # запросов
        self.per_seconds = per_seconds  # в течение периода
        self.tokens = rate_limit
        self.last_refill = time.time()

    async def acquire(self):
        # Пополняем токены пропорционально прошедшему времени
        now = time.time()
        elapsed = now - self.last_refill
        refill = elapsed * (self.rate_limit / self.per_seconds)
        self.tokens = min(self.rate_limit, self.tokens + refill)
        self.last_refill = now

        if self.tokens < 1:
            # Недостаточно токенов, ждем
            wait_time = (1 - self.tokens) * (self.per_seconds / self.rate_limit)
            logger.info(f"Rate limit: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.tokens = 1

        self.tokens -= 1
        return