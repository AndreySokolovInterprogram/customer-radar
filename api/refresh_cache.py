"""Периодически рассчитывает статистику по отраслям и городам из
исходной базы данных и сохраняет результат в локальное хранилище данных cache.db,
которое читает main.py.

Безопасность:
Перед началом работы скрипт проверяет доступность PostgreSQL и наличие данных.
Если база данных недоступна или остановлена, скрипт безопасно завершает работу (sys.exit(0)),
не затрагивая и не затирая существующие файлы данных.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import asyncpg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("customer_radar_refresh")

DATABASE_URL = os.environ.get("RADAR_DATABASE_URL", "")
CACHE_DB_PATH = Path(__file__).parent / "cache.db"
REFRESH_INTERVAL_SECONDS = int(os.environ.get("RADAR_REFRESH_INTERVAL_SECONDS", "900"))

SUBJECTS = {
    "customer_acquisition": "Маркетинг и привлечение клиентов (B2B)",
    "tender_support": "Сопровождение тендеров и госзакупок",
    "food_delivery": "Доставка еды и готовых блюд",
    "food_service": "Общепит, кафе и рестораны",
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "create table if not exists cache ("
        "key text primary key, payload text not null, computed_at text not null)"
    )
    conn.commit()


async def compute_subject(pool: asyncpg.Pool, subject: str, city: Optional[str]) -> dict:
    city_filter = "and c.city->>'title' = $2" if city else ""
    args = [subject] + ([city] if city else [])

    summary = await pool.fetchrow(
        f"""
        select
          count(distinct gc.id) as groups_total,
          count(gp.id) as posts_total,
          coalesce(sum(gp.likes_count), 0) as likes_total,
          coalesce(sum(gp.reposts_count), 0) as reposts_total,
          coalesce(sum(gp.comments_count), 0) as comments_total
        from group_candidates gc
        join group_labels gl on gl.group_id = gc.id and gl.label = $1
        join vk_communities c on c.vk_id = gc.vk_id
        left join group_posts gp on gp.group_id = gc.id
        where gc.classification_status = 'approved' {city_filter}
        """,
        *args,
    )

    top_posts = await pool.fetch(
        f"""
        select
          gc.name as group_name,
          c.city->>'title' as city,
          gp.text as text,
          gp.likes_count, gp.reposts_count, gp.comments_count,
          (gp.likes_count + gp.reposts_count + gp.comments_count) as engagement,
          gp.published_at, gp.vk_owner_id, gp.vk_post_id
        from group_candidates gc
        join group_labels gl on gl.group_id = gc.id and gl.label = $1
        join vk_communities c on c.vk_id = gc.vk_id
        join group_posts gp on gp.group_id = gc.id
        where gc.classification_status = 'approved' {city_filter}
        order by engagement desc
        limit 5
        """,
        *args,
    )

    return {
        "subject": subject,
        "subject_title": SUBJECTS[subject],
        "city": city,
        "groups_total": summary["groups_total"] if summary else 0,
        "posts_total": summary["posts_total"] if summary else 0,
        "likes_total": summary["likes_total"] if summary else 0,
        "reposts_total": summary["reposts_total"] if summary else 0,
        "comments_total": summary["comments_total"] if summary else 0,
        "top_posts": [
            {
                "group_name": row["group_name"],
                "city": row["city"],
                "text": (row["text"] or "")[:280],
                "likes": row["likes_count"],
                "reposts": row["reposts_count"],
                "comments": row["comments_count"],
                "engagement": row["engagement"],
                "published_at": row["published_at"].isoformat() if row["published_at"] else None,
                "url": f"https://vk.com/wall{row['vk_owner_id']}_{row['vk_post_id']}",
            }
            for row in top_posts
        ],
    }


async def refresh_once(pool: asyncpg.Pool, conn: sqlite3.Connection) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    cities = [
        row["city"]
        for row in await pool.fetch(
            "select distinct city->>'title' as city from vk_communities "
            "where city is not null and city->>'title' is not null order by 1"
        )
    ]
    conn.execute(
        "insert into cache (key, payload, computed_at) values ('__cities__', ?, ?) "
        "on conflict(key) do update set payload=excluded.payload, computed_at=excluded.computed_at",
        (json.dumps(cities, ensure_ascii=False), now),
    )
    conn.commit()

    for subject in SUBJECTS:
        payload = await compute_subject(pool, subject, None)
        conn.execute(
            "insert into cache (key, payload, computed_at) values (?, ?, ?) "
            "on conflict(key) do update set payload=excluded.payload, computed_at=excluded.computed_at",
            (subject, json.dumps(payload, ensure_ascii=False), now),
        )
        conn.commit()
        logger.info("[refresh] %s: %s групп, %s постов", subject, payload["groups_total"], payload["posts_total"])

        for city in cities:
            city_payload = await compute_subject(pool, subject, city)
            if city_payload["groups_total"] == 0:
                continue
            key = f"{subject}::{city}"
            conn.execute(
                "insert into cache (key, payload, computed_at) values (?, ?, ?) "
                "on conflict(key) do update set payload=excluded.payload, computed_at=excluded.computed_at",
                (key, json.dumps(city_payload, ensure_ascii=False), now),
            )
            conn.commit()


async def verify_postgres_safe() -> Optional[asyncpg.Pool]:
    """Безопасная проверка подключения к PostgreSQL перед открытием локального хранилища."""
    if not DATABASE_URL:
        logger.info("[safe-guard] RADAR_DATABASE_URL не задана. Локальные данные сохранены без изменений.")
        return None

    try:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2, command_timeout=10, timeout=5)
    except Exception as exc:
        logger.info("[safe-guard] PostgreSQL недоступен или остановлен (%s). Локальные данные сохранены без изменений.", exc)
        return None

    try:
        async with pool.acquire() as conn:
            cnt = await conn.fetchval(
                "select count(*) from group_candidates where classification_status = 'approved'"
            )
            if not cnt or cnt == 0:
                logger.info("[safe-guard] В базе данных 0 подтверждённых групп. Локальные данные сохранены без изменений.")
                await pool.close()
                return None
    except Exception as exc:
        logger.info("[safe-guard] Проверочный запрос завершился ошибкой (%s). Локальные данные сохранены без изменений.", exc)
        await pool.close()
        return None

    return pool


async def main() -> None:
    pool = await verify_postgres_safe()
    if pool is None:
        logger.info("[safe-guard] Завершение работы без изменения локального хранилища.")
        sys.exit(0)

    conn = sqlite3.connect(CACHE_DB_PATH)
    init_db(conn)
    try:
        while True:
            started = time.monotonic()
            try:
                await refresh_once(pool, conn)
                logger.info("[refresh] Обновление завершено за %.1fс", time.monotonic() - started)
            except Exception as error:  # noqa: BLE001
                logger.warning("[refresh] Ошибка обновления: %s", error)
            await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
    finally:
        await pool.close()
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
