"""Read-only API для сервиса «Проект Маркли» (Customer Radar).

Отдаёт агрегированную аналитику по отраслям и городам,
а также публикации с максимальной вовлечённостью.
Работает автономно с локальным хранилищем данных.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("customer_radar_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_DB_PATH = Path(__file__).parent / "cache.db"

# Защищённый ключ приложения VK (ID 54674554)
VK_SECRET_KEY = os.environ.get("VK_SECRET_KEY") or os.environ.get("VK_SECURE_KEY") or ""
VK_SIGN_REQUIRED = os.environ.get("VK_SIGN_REQUIRED", "true").lower() in ("true", "1", "yes")

SUBJECTS = {
    "customer_acquisition": {
        "title": "Маркетинг и привлечение клиентов (B2B)",
        "description": "B2B-аудитория, маркетинг, реклама и лидогенерация для малого и среднего бизнеса",
    },
    "tender_support": {
        "title": "Сопровождение тендеров и госзакупок",
        "description": "Поставщики, участники торгов, специалисты по 44-ФЗ и 223-ФЗ",
    },
    "food_delivery": {
        "title": "Доставка еды и готовых блюд",
        "description": "Сервисы доставки, кухни, локальные сети питания",
    },
    "food_service": {
        "title": "Общепит, кафе и рестораны",
        "description": "Заведения общественного питания, кофейни и ресторанный бизнес",
    },
}

app = FastAPI(title="Проект Маркли — Аналитика аудитории и сообществ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vk.markly.ru",
        "https://vk.com",
        "https://m.vk.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def verify_vk_sign(launch_params_raw: str, secret_key: str) -> bool:
    """Проверяет подпись sign параметров запуска VK Mini Apps по стандарту HMAC-SHA256."""
    if not launch_params_raw or not secret_key:
        return False
    try:
        # Отрезаем ведущий знак вопроса, если передан
        if launch_params_raw.startswith("?"):
            launch_params_raw = launch_params_raw[1:]

        parsed = urllib.parse.parse_qsl(launch_params_raw, keep_blank_values=True)
        sign_from_params = None
        vk_params = []

        for k, v in parsed:
            if k == "sign":
                sign_from_params = v
            elif k.startswith("vk_"):
                vk_params.append((k, v))

        if not sign_from_params or not vk_params:
            return False

        # Сортировка по ключу в лексикографическом порядке
        vk_params.sort(key=lambda item: item[0])
        encoded_params = urllib.parse.urlencode(vk_params, doseq=True)

        digest = hmac.new(
            secret_key.encode("utf-8"),
            encoded_params.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        calculated_sign = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        return hmac.compare_digest(calculated_sign, sign_from_params)
    except Exception as exc:
        logger.warning("Ошибка при проверке подписи параметров VK: %s", exc)
        return False


def check_request_sign(request: Request) -> None:
    """Проверяет подпись параметров запуска в заголовке или строке запроса."""
    if not VK_SECRET_KEY:
        if VK_SIGN_REQUIRED:
            logger.warning("VK_SECRET_KEY не задан в переменных окружения. Проверка подписи пропущена.")
        return

    launch_params_raw = request.headers.get("X-VK-Launch-Params") or ""
    if not launch_params_raw and request.url.query:
        # Проверяем, есть ли параметры vk_ в строке запроса
        if "vk_" in request.url.query:
            launch_params_raw = request.url.query

    if not launch_params_raw:
        if VK_SIGN_REQUIRED:
            raise HTTPException(
                status_code=403,
                detail="Доступ запрещён: отсутствуют параметры запуска мини-приложения",
            )
        return

    if not verify_vk_sign(launch_params_raw, VK_SECRET_KEY):
        raise HTTPException(
            status_code=403,
            detail="Доступ запрещён: некорректная подпись параметров запуска",
        )


def _read_data_raw(key: str) -> Optional[object]:
    """Быстрое чтение готовых аналитических срезов из локальной базы данных."""
    if not CACHE_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{CACHE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute("select payload from cache where key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row is None or not row[0]:
            return None
        return json.loads(row[0])
    except Exception as exc:
        logger.error("Ошибка чтения данных для ключа %s: %s", key, exc)
        return None


@app.get("/api/subjects")
async def subjects(request: Request):
    """Список поддерживаемых отраслей бизнеса."""
    check_request_sign(request)
    return [
        {
            "key": key,
            "title": info["title"],
            "description": info["description"],
        }
        for key, info in SUBJECTS.items()
    ]


@app.get("/api/cities")
async def cities(request: Request):
    """Список доступных городов с собранной аналитикой."""
    check_request_sign(request)
    data = _read_data_raw("__cities__")
    return data if isinstance(data, list) else []


@app.get("/api/stats")
async def stats(
    request: Request,
    subject: str = Query(..., description="Ключ предметной области"),
    city: Optional[str] = Query(None, description="Фильтр по городу (точное совпадение)"),
):
    """Аналитическая сводка и публикации с максимальной вовлечённостью."""
    check_request_sign(request)

    if subject not in SUBJECTS:
        raise HTTPException(status_code=400, detail=f"Неизвестная отрасль: {subject}")

    key = f"{subject}::{city}" if city else subject
    payload = _read_data_raw(key)

    if payload is None or not isinstance(payload, dict):
        if city:
            raise HTTPException(
                status_code=404,
                detail=f"По направлению «{SUBJECTS[subject]['title']}» для города «{city}» пока недостаточно данных для построения аналитики",
            )
        raise HTTPException(
            status_code=404,
            detail="Данные по выбранному направлению формируются, пожалуйста, обновите страницу через минуту",
        )

    # Актуализируем человекочитаемое наименование отрасли
    payload["subject_title"] = SUBJECTS[subject]["title"]
    if city:
        payload["city"] = city

    # Удаляем технические поля при их наличии
    payload.pop("cached_at", None)
    payload.pop("computed_at", None)

    return payload
