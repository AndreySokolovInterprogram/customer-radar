"""Read-only API для Клиентского Радара (VK Mini App).

Отдаёт агрегаты по предметным областям NCA-RS и топ-5 постов по вовлечённости
(лайки+репосты+комментарии), с опциональным фильтром по городу. Читает из
локального SQLite-кэша (см. refresh_cache.py) — прямые агрегатные запросы
к ~1.3М постов на каждый HTTP-запрос были неприемлемо медленными (см. коммит),
поэтому фоновый процесс пересчитывает кэш раз в 15 минут отдельно.
"""

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

CACHE_DB_PATH = Path(__file__).parent / "cache.db"

SUBJECTS = {
    "food_delivery": "Доставка еды",
    "food_service": "Общепит и кафе",
    "customer_acquisition": "Привлечение клиентов",
    "tender_support": "Тендерное сопровождение",
}

app = FastAPI(title="Клиентский Радар API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vk.markly.ru", "https://vk.com"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _read_cache_raw(key: str) -> tuple[object, str] | None:
    conn = sqlite3.connect(CACHE_DB_PATH)
    try:
        row = conn.execute("select payload, computed_at from cache where key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return json.loads(row[0]), row[1]


def _read_cache(key: str) -> dict | None:
    raw = _read_cache_raw(key)
    if raw is None:
        return None
    payload, computed_at = raw
    payload["cached_at"] = computed_at
    return payload


@app.get("/api/subjects")
async def subjects():
    return [{"key": key, "title": title} for key, title in SUBJECTS.items()]


@app.get("/api/cities")
async def cities():
    raw = _read_cache_raw("__cities__")
    return raw[0] if raw is not None else []


@app.get("/api/stats")
async def stats(
    subject: str = Query(..., description="Ключ предметной области"),
    city: str | None = Query(None, description="Фильтр по городу (точное совпадение)"),
):
    if subject not in SUBJECTS:
        raise HTTPException(400, f"Неизвестная область: {subject}")
    key = f"{subject}::{city}" if city else subject
    payload = _read_cache(key)
    if payload is None:
        if city:
            raise HTTPException(
                404, f"Нет данных для «{SUBJECTS[subject]}» + город «{city}» (мало сообществ с этим городом пока собрано)"
            )
        raise HTTPException(503, "Кэш ещё не посчитан, попробуйте через минуту")
    return payload
