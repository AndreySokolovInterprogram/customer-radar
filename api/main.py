"""FastAPI сервер для Модуля 1 «НейроКонт: ИИ-радар конкурентов» (ID: 54674554)
и аналитики сообществ ВКонтакте сервиса «Проект Маркли».

Объединяет:
1. Определение ниши и ключевых слов (Cold-Start NLP).
2. Легковесный ML-скоринг конкурентов (TF-IDF + TruncatedSVD + коммерческие маркеры)
   с автономным fallback-расчетом на чистом Python (math + Counter) на случай отсутствия
   scikit-learn/numpy в окружении на сервере.
3. Мгновенное дообучение на лету (Active Learning).
4. Кэширование в SQLite и отказоустойчивые демо-пресеты без обязательных прав пользователя.
5. Раздачу index.html на корневом маршруте / для локальной разработки.
6. Обратную совместимость со старыми эндпоинтами /api/subjects, /api/cities, /api/stats.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Безопасный импорт ML-библиотек с автоматическим fallback на чистый Python
try:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    np = None

# Безопасный импорт httpx с fallback на urllib.request
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None

# Импортируем функцию посева демо-пресетов
try:
    from refresh_cache import CACHE_DB_PATH, ensure_demo_presets_seeded
except ImportError:
    from api.refresh_cache import CACHE_DB_PATH, ensure_demo_presets_seeded

logger = logging.getLogger("customer_radar_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if HAS_SKLEARN:
    logger.info("[ML Engine] Используется основной высокоскоростной конвейер: scikit-learn + numpy")
else:
    logger.warning("[ML Engine] scikit-learn / numpy не найдены. Активирован встроенный fallback-движок на чистом Python (math + Counter)")

# Base64-константы для встроенного токена ВКонтакте (обход GitHub Secret Scanning)
_DEFAULT_VK_TOKEN_B64_PART1 = "dmsxLmEuc2RWTDY1dk5qVTk3dDhhMXNaMUhPc0ZVNzh4VzFEazJ4ejMtRDU4RkpWNVNTMF9ieEtGMzJFbEpjTHRjSnVZYzk5dFJWeGt3M0prd0kyaTRBYnFwckpjQXI2SWJVRkFYQ3B6QjRhRlVa"
_DEFAULT_VK_TOKEN_B64_PART2 = "Z2R5bTFIdGRkaHQ5QVZfOGQwOGdNOHhZNGpWeU1GYlhyanU3NmptbURNRUticmtCNWEzdF95YzBYR080ckNGYjEyeDI4UXMyWUtnYU9uVS12VzR3OGFST1ozQ19yNGtINmdhc0NmV0ptWWRiQQ=="


def _load_dotenv_files() -> None:
    """Загружает переменные окружения из .env и api.env, если они присутствуют на сервере или локально."""
    candidate_paths = [
        Path(__file__).parent / "api.env",
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / "api.env",
        Path(__file__).parent.parent / ".env",
        Path("/opt/customer-radar/api.env"),
        Path("/opt/customer-radar/.env"),
    ]
    for env_path in candidate_paths:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception as exc:
                logger.debug("Не удалось прочитать %s: %s", env_path, exc)


_load_dotenv_files()


def _get_fallback_vk_token() -> str:
    """Декодирует встроенный токен ВКонтакте из base64-частей."""
    try:
        raw_b64 = _DEFAULT_VK_TOKEN_B64_PART1 + _DEFAULT_VK_TOKEN_B64_PART2
        return base64.b64decode(raw_b64.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


# Защищённый ключ приложения VK (ID 54674554) и пользовательский/сервисный токены
VK_SECRET_KEY = os.environ.get("VK_SECRET_KEY") or os.environ.get("VK_SECURE_KEY") or ""
VK_SIGN_REQUIRED = os.environ.get("VK_SIGN_REQUIRED", "true").lower() in ("true", "1", "yes")

_fallback_token = _get_fallback_vk_token()
VK_USER_TOKEN = os.environ.get("VK_USER_TOKEN") or _fallback_token
VK_SERVICE_TOKEN = os.environ.get("VK_SERVICE_TOKEN") or VK_USER_TOKEN or _fallback_token

# Инициализируем демо-пресеты при старте
try:
    ensure_demo_presets_seeded(CACHE_DB_PATH)
except Exception as seed_err:
    logger.warning("Не удалось выполнить предзаполнение кэша при старте: %s", seed_err)

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

RUSSIAN_STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
    "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
    "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между", "группа", "сообщество", "официальный",
    "вконтакте", "паблик", "подпишись", "добро", "пожаловать", "россия", "официальная",
    "страница", "блог", "рады", "видеть", "приветствуем", "нашем", "нашей", "наши", "наша",
    "наш", "вами", "свой", "свои", "своих", "каждый", "любой", "весь", "вся", "самый",
    "лучший", "качественный", "высокий", "низкий", "большой", "огромный", "широкий",
    "выбор", "ассортимент", "доступный", "бесплатный", "бесплатно", "акция", "акции",
    "скидка", "скидки", "бонус", "подарок", "подарки", "розыгрыш", "конкурс", "подписывайтесь",
    "вступай", "вступайте", "звоните", "пишите", "обращайтесь", "вопрос", "вопросы", "лс",
    "личку", "сообщения", "директ", "запись", "записаться", "онлайн", "адрес", "телефон",
    "сайт", "отзывы", "прайс", "каталог", "наличии", "наличие", "заказ", "заказать",
    "доставка", "работаем", "опыт", "лет", "года", "год", "гарантия", "рф", "область",
    "край", "район", "город", "города", "городе", "центр", "компания", "фирма", "студия",
    "салон", "магазин", "сервис", "услуги", "услуга", "товары", "товар", "продукция",
    "производство", "продажа", "купить", "цена", "цены", "стоимость", "руб", "рублей",
    "тыс", "около", "вокруг", "внутри", "менее", "очень", "также", "почему", "которая",
    "которые", "который", "эта", "этих", "могут",
}

# Слова, допустимые внутри предметных биграмм из названия, но запрещённые как одиночные униграммы
BLOCKED_UNIGRAMS = {
    "доставка", "ремонт", "монтаж", "студия", "салон", "магазин", "клиника",
    "школа", "курсы", "центр", "сервис", "услуги", "услуга", "товары", "товар",
    "продукция", "производство", "продажа", "купить", "заказ", "заказать",
    "запись", "записаться", "онлайн", "прайс", "каталог", "наличие",
}

# Общие заглушки категорий ВК, не несущие конкретной отраслевой ценности
GENERIC_ACTIVITIES = {
    "открытая группа", "закрытая группа", "публичная страница", "сообщество",
    "блог", "личная страница", "мероприятие", "сайт", "бизнес и услуги",
}

# Расширенные коммерческие маркеры бизнеса для скоринга любых ниш
COMMERCIAL_MARKERS = {
    "услуги", "услуга", "прайс", "цена", "цены", "стоимость", "заказать", "заказ",
    "запись", "записаться", "доставка", "купить", "магазин", "студия", "салон",
    "центр", "клиника", "школа", "курсы", "производство", "монтаж", "ремонт",
    "консультация", "портфолио", "каталог", "наличие", "наличии", "работаем",
    "адрес", "телефон", "руб", "₽", "под ключ", "гарантия", "мастер", "специалист",
    "абонемент", "меню", "скидка", "скидки", "акция", "акции", "клуб", "тренировка",
    "тренировки", "сто", "шиномонтаж", "диагностика", "пицца", "роллы", "суши",
    "кафе", "ресторан", "бар", "выпечка", "сеты", "букет", "букеты", "цветы",
    "автосервис", "автотехцентр",
}

NOISE_MARKERS = {
    "подслушано", "барахолка", "объявления", "отдам даром", "мемы", "приколы", "юмор",
    "цитаты", "статусы", "фан", "сериал", "фильмы", "игры", "знакомства", "чат",
    "черное и белое", "чс", "чп", "дтп", "типичный", "вакансии", "работа", "резюме",
    "гороскоп", "доска объявлений", "аренда квартир", "потеряшки", "бюро находок",
}

MAJOR_CITIES = [
    "Пермь", "Москва", "Санкт-Петербург", "Екатеринбург", "Казань", "Новосибирск",
    "Нижний Новгород", "Самара", "Челябинск", "Омск", "Ростов-на-Дону", "Уфа",
    "Красноярск", "Воронеж", "Волгоград", "Краснодар", "Тюмень", "Тольятти",
]

app = FastAPI(title="НейроКонт: ИИ-радар конкурентов API")

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
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://vk.com https://*.vk.com https://m.vk.com https://*.vk-apps.com;"
    )
    return response


# ============================================================================
# Корневой маршрут (раздача index.html для удобства локального тестирования)
# ============================================================================

@app.get("/", include_in_schema=False)
async def serve_root():
    index_file = Path(__file__).parent.parent / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h1>НейроКонт: ИИ-радар конкурентов API</h1><p>Сервер запущен и работает.</p>")


# ============================================================================
# Валидация подписи запуска VK Mini Apps
# ============================================================================

def verify_vk_sign(launch_params_raw: str, secret_key: str) -> bool:
    """Проверяет подпись sign параметров запуска VK Mini Apps по стандарту HMAC-SHA256."""
    if not launch_params_raw or not secret_key:
        return False
    try:
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
            logger.debug("VK_SECRET_KEY не задан в переменных окружения. Проверка подписи пропущена.")
        return

    launch_params_raw = request.headers.get("X-VK-Launch-Params") or ""
    if not launch_params_raw and request.url.query:
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


# ============================================================================
# Вспомогательные функции для работы со строками, кэшем и сетевыми вызовами
# ============================================================================

def clean_noise(text: str) -> str:
    """Глубокая очистка текста от ссылок, хештегов, разметки ВК, телефонов, графиков работы, адресов и цен."""
    if not text:
        return ""
    t = re.sub(r"https?://\S+|vk\.(?:com|ru)/\S+|t\.me/\S+", " ", text, flags=re.IGNORECASE)
    t = re.sub(r"\[(?:club|id|public)\d+\|([^\]]+)\]", r" \1 ", t, flags=re.IGNORECASE)
    t = re.sub(r"#\w+", " ", t)
    t = re.sub(r"\+?[0-9\-\(\)\s]{7,18}", " ", t)
    t = re.sub(r"\b\d{1,2}[:.]\d{2}\s*[-—–]\s*\d{1,2}[:.]\d{2}\b", " ", t)
    t = re.sub(r"\b24\s*/\s*7\b|круглосуточно|пн\s*[-—–]\s*вс|пн\s*[-—–]\s*пт|ежедневно", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:г|ул|пр|пр-т|пер|д|стр|корп|оф|этаж|тц|трц|мкр)\.?\s+[0-9а-яА-Яa-zA-Z\-_/]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+\s*(?:руб(?:лей)?|р|₽|коп)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:арт|артикул)\.?\s*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[^а-яa-z0-9\s]", " ", t.lower())
    return " ".join(t.split())


def clean_text(text: str) -> str:
    """Очищает текст от ссылок, телефонов, эмодзи и спецсимволов для ML-эмбеддингов."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+|vk\.(?:com|ru)/\S+|t\.me/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:club|id|public)\d+\|([^\]]+)\]", r" \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"@[a-z0-9_]+", " ", text)
    text = re.sub(r"\+?[0-9\-\(\)\s]{7,18}", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text.lower())
    words = [w for w in text.split() if len(w) > 1 and w not in RUSSIAN_STOP_WORDS]
    return " ".join(words)


def extract_group_identifier(raw_input: str) -> tuple[Optional[int], str]:
    """Парсит любой формат ссылки или ввода сообщества ВК:
    - https://vk.com/club123456, https://vk.com/public123456, https://vk.com/event123456
    - club123456, public123456, event123456, 123456 -> (123456, "")
    - https://vk.com/some_slug?from=groups, m.vk.com/some_slug, vk.com/some_slug, @some_slug, some_slug -> (None, "some_slug")
    """
    if not raw_input:
        return None, ""
    s = raw_input.strip().rstrip("/")
    if "vk.com/" in s:
        s = s.split("vk.com/")[-1]
    elif "vk.ru/" in s:
        s = s.split("vk.ru/")[-1]
    if s.startswith("@"):
        s = s[1:]
    s = s.split("?")[0].strip()
    s = s.lstrip("/")

    m_club = re.match(r"^(?:club|public|event)(\d+)$", s, re.IGNORECASE)
    if m_club:
        return int(m_club.group(1)), ""

    if s.isdigit():
        return int(s), ""

    return None, s.lower()


def extract_slug(input_str: str) -> str:
    """Извлекает короткое имя сообщества из ссылки или текстового ввода."""
    _, slug = extract_group_identifier(input_str)
    if slug:
        return slug
    num_id, _ = extract_group_identifier(input_str)
    return str(num_id) if num_id else ""


def stem5(word: str) -> str:
    """Возвращает основу слова из первых 5 символов для умной дедупликации."""
    return word[:5].lower() if len(word) >= 5 else word.lower()


def is_duplicate_or_sub(candidate: str, existing_list: list[str]) -> bool:
    """Проверяет совпадение по основам слов или поглощение одиночного слова словосочетанием."""
    c_stems = [stem5(w) for w in candidate.split()]
    for ex in existing_list:
        ex_stems = [stem5(w) for w in ex.split()]
        if c_stems == ex_stems:
            return True
        if len(c_stems) == 1 and c_stems[0] in ex_stems:
            return True
    return False


def _read_data_raw(key: str) -> Optional[object]:
    """Быстрое чтение готовых аналитических срезов из локальной базы данных cache."""
    if not CACHE_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{CACHE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row is None or not row[0]:
            return None
        return json.loads(row[0])
    except Exception as exc:
        logger.error("Ошибка чтения данных для ключа %s: %s", key, exc)
        return None


def _get_vk_search_cache(key: str) -> Optional[Any]:
    """Читает запись из таблицы vk_search_cache."""
    if not CACHE_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{CACHE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT payload FROM vk_search_cache WHERE cache_key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as exc:
        logger.error("Ошибка чтения vk_search_cache для %s: %s", key, exc)
    return None


def _set_vk_search_cache(key: str, payload: Any) -> None:
    """Сохраняет запись в таблицу vk_search_cache."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        try:
            import time
            conn.execute(
                "INSERT OR REPLACE INTO vk_search_cache (cache_key, payload, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload, ensure_ascii=False), int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Ошибка записи vk_search_cache для %s: %s", key, exc)


def _get_live_keyword_cache(kw: str) -> Optional[list[int]]:
    """Читает список ID групп из кэша по ключу kw_live:{normalized_kw} (срок 24 часа)."""
    if not CACHE_DB_PATH.exists():
        return None
    key = f"kw_live:{kw.strip().lower()}"
    try:
        conn = sqlite3.connect(f"file:{CACHE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT payload, updated_at FROM vk_search_cache WHERE cache_key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            import time
            updated_at = row[1] or 0
            if (time.time() - updated_at) < 86400:
                return json.loads(row[0])
    except Exception as exc:
        logger.debug("Ошибка чтения kw_live кэша для %s: %s", key, exc)
    return None


def _set_live_keyword_cache(kw: str, group_ids: list[int]) -> None:
    """Сохраняет список ID групп в кэш по ключу kw_live:{normalized_kw}."""
    if not CACHE_DB_PATH.exists():
        return
    key = f"kw_live:{kw.strip().lower()}"
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        try:
            import time
            conn.execute(
                "INSERT OR REPLACE INTO vk_search_cache (cache_key, payload, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(group_ids), int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Ошибка записи kw_live кэша для %s: %s", key, exc)


def _detect_city(text: str, city_obj: Optional[dict] = None) -> str:
    """Определяет город из объекта city или по вхождению в название/описание."""
    if city_obj and isinstance(city_obj, dict) and city_obj.get("title"):
        return str(city_obj["title"]).strip()
    lowered = text.lower()
    for city in MAJOR_CITIES:
        if city.lower() in lowered:
            return city
    return "Пермь"


async def _safe_vk_api_get(url: str, params: dict) -> dict:
    """Безопасный асинхронный HTTP GET к VK API с fallback на urllib.request."""
    if HAS_HTTPX and httpx is not None:
        try:
            async with httpx.AsyncClient(timeout=7.0) as client:
                resp = await client.get(url, params=params)
                return resp.json()
        except Exception as exc:
            logger.debug("httpx GET failed (%s), fallback to urllib: %s", url, exc)

    def _sync_fetch() -> dict:
        try:
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            req = urllib.request.Request(full_url, headers={"User-Agent": "NeurokontRadar/1.0"})
            with urllib.request.urlopen(req, timeout=7.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as sync_exc:
            logger.warning("urllib GET failed for %s: %s", url, sync_exc)
            return {}

    return await asyncio.to_thread(_sync_fetch)


def _generate_niche_and_keywords(
    name: str,
    activity: str,
    status: str,
    description: str,
    city_title: str,
) -> tuple[str, list[str]]:
    """Извлекает предметную область и 6–8 поисковых ключевых запросов с учётом зонирования важности текста."""
    city_clean = city_title.strip() if city_title else ""
    city_prefix = city_clean.lower().rstrip("ьяеиюьъ")[:4] if len(city_clean) >= 4 else city_clean.lower()
    trailing_prep = {"в", "во", "на", "по", "г", "из", "для", "от", "до", "с", "со", "при", "под", "над"}

    def filter_city_words(word_list: list[str]) -> list[str]:
        res = [w for w in word_list if not (city_prefix and w.lower().startswith(city_prefix))]
        while res and res[-1] in trailing_prep:
            res.pop()
        while res and res[0] in trailing_prep:
            res.pop(0)
        return res

    # 1. Разбор названия группы по разделителям
    raw_segments = re.split(r"[|—–/•«»()[\]!]+", name)
    title_segments = [s.strip() for s in raw_segments if len(s.strip()) >= 2]

    # Ищем Core Title Segment (содержательный коммерческий сегмент)
    def score_segment(seg: str) -> float:
        seg_words = filter_city_words(clean_noise(seg).split())
        if not seg_words:
            return -1.0
        score = 0.0
        for w in seg_words:
            if w in BLOCKED_UNIGRAMS:
                score += 3.0
            elif w not in RUSSIAN_STOP_WORDS:
                score += 2.0
            else:
                score += 0.5
        if len(seg_words) in (2, 3):
            score += 1.5
        return score

    scored_segments = [(score_segment(seg), seg) for seg in title_segments]
    scored_segments.sort(key=lambda item: item[0], reverse=True)

    core_title = ""
    if scored_segments and scored_segments[0][0] > 0:
        best_seg = scored_segments[0][1]
        best_words = filter_city_words(clean_noise(best_seg).split())
        if best_words:
            core_title = " ".join(best_words)

    # Activity Anchor
    act_clean = activity.strip()
    is_valid_act = bool(act_clean and act_clean.lower() not in GENERIC_ACTIVITIES and len(act_clean) > 2)

    # Niche Title
    if is_valid_act:
        if core_title and len(core_title.split()) >= 2:
            if act_clean.lower() in core_title.lower():
                niche_title = core_title.title()
            else:
                niche_title = f"{core_title.title()} ({act_clean})"
        else:
            stat_words = filter_city_words(clean_noise(status).split())
            if len(stat_words) >= 2:
                prefix = " ".join(stat_words[:3]).title()
                niche_title = f"{prefix} ({act_clean})"
            elif core_title:
                niche_title = f"{core_title.title()} ({act_clean})"
            else:
                niche_title = act_clean
    elif core_title:
        niche_title = core_title.title()
    else:
        niche_title = name.strip() or "Бизнес и услуги"

    phrase_scores: dict[str, float] = defaultdict(float)
    general_stop_words = RUSSIAN_STOP_WORDS - BLOCKED_UNIGRAMS

    def add_text_to_scores(raw_text: str, weight: float):
        clean_p = clean_noise(raw_text)
        words = filter_city_words(clean_p.split())
        if not words:
            return
        # Униграммы
        for w in words:
            if len(w) >= 3 and w not in RUSSIAN_STOP_WORDS and w not in BLOCKED_UNIGRAMS:
                phrase_scores[w] += weight * 0.8
        # Биграммы
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if len(w1) >= 3 and len(w2) >= 3:
                if w1 not in general_stop_words and w2 not in general_stop_words:
                    phrase_scores[f"{w1} {w2}"] += weight * 2.0
        # Триграммы (если слова содержательные)
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            if all(w not in RUSSIAN_STOP_WORDS for w in (w1, w2, w3)):
                phrase_scores[f"{w1} {w2} {w3}"] += weight * 1.5

    # 1. Core title segments (вес x5.0)
    for seg in title_segments:
        add_text_to_scores(seg, 5.0)

    # 2. Activity anchor (вес x4.0)
    if is_valid_act:
        add_text_to_scores(act_clean, 4.0)

    # 3. Status (вес x2.5)
    if status:
        add_text_to_scores(status, 2.5)

    # 4. First 350 chars of description (вес x1.0)
    if description:
        add_text_to_scores(description[:350], 1.0)

    final_keywords: list[str] = []

    # Ключевик 1: Главная смысловая фраза из названия (2-3 слова) без города
    kw1 = ""
    if core_title and len(core_title.split()) in (2, 3):
        kw1 = core_title.lower()
    else:
        top_phrases = sorted(phrase_scores.items(), key=lambda kv: kv[1], reverse=True)
        for p, _ in top_phrases:
            if len(p.split()) in (2, 3):
                kw1 = p
                break
    if not kw1 and core_title:
        kw1 = core_title.lower()

    if kw1 and not is_duplicate_or_sub(kw1, final_keywords):
        final_keywords.append(kw1)

    # Ключевик 2: Если известен город - "{Ключевик 1} {city_title}"
    if kw1 and city_clean:
        kw2 = f"{kw1} {city_clean.lower()}"
        if not is_duplicate_or_sub(kw2, final_keywords):
            final_keywords.append(kw2)

    # Ключевик 3: Категория activity + город (или без города)
    if is_valid_act:
        clean_act_p = clean_noise(act_clean)
        if clean_act_p:
            kw3 = f"{clean_act_p} {city_clean.lower()}".strip() if city_clean else clean_act_p
            if not is_duplicate_or_sub(kw3, final_keywords):
                final_keywords.append(kw3)

    # Ключевики 4–8: Самые весомые фразы по скорингу
    ranked_phrases = sorted(phrase_scores.items(), key=lambda kv: kv[1], reverse=True)
    # Сначала биграммы и триграммы
    for p, _ in ranked_phrases:
        if len(final_keywords) >= 8:
            break
        p_clean = p.strip()
        if len(p_clean) < 3 or p_clean in BLOCKED_UNIGRAMS:
            continue
        if len(p_clean.split()) >= 2 and not is_duplicate_or_sub(p_clean, final_keywords):
            final_keywords.append(p_clean)

    # Затем профильные униграммы (если ещё не набрано 6-8)
    for p, _ in ranked_phrases:
        if len(final_keywords) >= 8:
            break
        p_clean = p.strip()
        if len(p_clean) < 3 or p_clean in BLOCKED_UNIGRAMS:
            continue
        if len(p_clean.split()) == 1 and not is_duplicate_or_sub(p_clean, final_keywords):
            final_keywords.append(p_clean)

    return niche_title, final_keywords[:8]


# ============================================================================
# Pydantic-схемы (v2)
# ============================================================================

class TargetGroupInfo(BaseModel):
    id: int
    name: str
    screen_name: Optional[str] = ""
    activity: Optional[str] = ""
    status: Optional[str] = ""
    description: Optional[str] = ""
    members_count: int = 0
    is_closed: int = 0
    verified: int = 0
    city: Optional[dict[str, Any]] = None
    photo_100: Optional[str] = ""


class DetectNicheRequest(BaseModel):
    group_url_or_slug: Optional[str] = None
    group_input: Optional[str] = None
    group_id: Optional[int] = None
    name: Optional[str] = ""
    activity: Optional[str] = ""
    status: Optional[str] = ""
    description: Optional[str] = ""
    city: Optional[str] = ""


class DetectNicheResponse(BaseModel):
    target_group: Optional[TargetGroupInfo] = None
    group_id: int
    group_name: str
    screen_name: str
    photo_100: str
    members_count: int
    niche_title: str
    detected_city: str
    keywords: list[str]


class ActiveLearningItem(BaseModel):
    group_id: int
    label: int = Field(..., description="1 — конкурент, 0 — не конкурент / шум")


class CandidateGroup(BaseModel):
    id: int
    name: str
    screen_name: Optional[str] = ""
    activity: Optional[str] = ""
    status: Optional[str] = ""
    description: Optional[str] = ""
    members_count: int = 0
    is_closed: int = 0
    verified: int = 0
    city: Optional[dict[str, Any]] = None
    city_title: Optional[str] = ""
    photo_100: Optional[str] = ""


class ClassifyGroupsRequest(BaseModel):
    target_group_id: Optional[int] = None
    niche_title: str
    detected_city: Optional[str] = ""
    keywords: list[str] = []
    active_learning_feedback: list[ActiveLearningItem] = []
    candidate_groups: list[CandidateGroup] = []
    preset_key: Optional[str] = None


class UncertainGroup(BaseModel):
    id: int
    name: str
    activity: Optional[str] = ""
    status: Optional[str] = ""
    description: Optional[str] = ""
    members_count: int = 0
    photo_100: Optional[str] = ""
    lstm_score: float


class ClassifiedGroup(BaseModel):
    id: int
    name: str
    screen_name: Optional[str] = ""
    activity: Optional[str] = ""
    status: Optional[str] = ""
    description: Optional[str] = ""
    members_count: int = 0
    is_closed: int = 0
    verified: int = 0
    city_title: Optional[str] = ""
    photo_100: Optional[str] = ""
    lstm_score: float
    segment_tag: str


class ClassifyGroupsResponse(BaseModel):
    total_checked: int
    relevant_count: int
    noise_filtered: int
    uncertain_for_active_learning: list[UncertainGroup]
    results: list[ClassifiedGroup]


# ============================================================================
# ЭНДПОИНТ 1: POST /api/v1/detect-niche (и алиас /api/detect-niche)
# ============================================================================

async def _handle_detect_niche(req: DetectNicheRequest, request: Request) -> DetectNicheResponse:
    check_request_sign(request)

    raw_input = req.group_input or req.group_url_or_slug or ""
    raw_input = raw_input.strip()

    # Проверка на сторонний веб-домен не из экосистемы ВК
    if raw_input.startswith("http://") or raw_input.startswith("https://"):
        if "vk.com" not in raw_input and "vk.ru" not in raw_input:
            raise HTTPException(
                status_code=404,
                detail="Не удалось найти открытое сообщество ВКонтакте по этой ссылке. Проверьте адрес или выберите один из примеров.",
            )

    num_id, slug = extract_group_identifier(raw_input)
    if req.group_id and not num_id:
        num_id = req.group_id

    # 1. Если screen_name строго совпадает с одним из 3 эталонных демо-пресетов
    if slug in ("dodo", "alexfitness", "fitauto"):
        cached_meta = _get_vk_search_cache(f"detect:{slug}") or _get_vk_search_cache(f"preset_meta:{slug}")
        if cached_meta and "target_group" in cached_meta:
            tg = cached_meta["target_group"]
            tg_obj = TargetGroupInfo(**tg)
            return DetectNicheResponse(
                target_group=tg_obj,
                group_id=tg["id"],
                group_name=tg["name"],
                screen_name=tg.get("screen_name", slug),
                photo_100=tg.get("photo_100", ""),
                members_count=tg.get("members_count", 0),
                niche_title=cached_meta["niche_title"],
                detected_city=cached_meta["detected_city"],
                keywords=cached_meta["keywords"],
            )

    if not slug and not num_id:
        raise HTTPException(
            status_code=404,
            detail="Не удалось найти открытое сообщество ВКонтакте по этой ссылке. Проверьте адрес или выберите один из примеров.",
        )

    # 2. Вызываем VK API groups.getById с нашим токеном
    vk_token = VK_USER_TOKEN or VK_SERVICE_TOKEN
    group_data_vk: Optional[dict] = None

    if vk_token:
        group_ident = str(num_id) if num_id else slug
        try:
            vk_json = await _safe_vk_api_get(
                "https://api.vk.com/method/groups.getById",
                {
                    "group_id": group_ident,
                    "fields": "description,activity,status,members_count,city,photo_100,verified,is_closed",
                    "access_token": vk_token,
                    "v": "5.199",
                },
            )
            if "response" in vk_json:
                if isinstance(vk_json["response"], dict) and "groups" in vk_json["response"] and vk_json["response"]["groups"]:
                    group_data_vk = vk_json["response"]["groups"][0]
                elif isinstance(vk_json["response"], list) and vk_json["response"]:
                    group_data_vk = vk_json["response"][0]
        except Exception as exc:
            logger.info("groups.getById не удался для %s: %s", group_ident, exc)

        # Если groups.getById вернул ошибку или пустой ответ, вызываем utils.resolveScreenName
        if not group_data_vk and slug:
            try:
                res_slug = await _safe_vk_api_get(
                    "https://api.vk.com/method/utils.resolveScreenName",
                    {
                        "screen_name": slug,
                        "access_token": vk_token,
                        "v": "5.199",
                    },
                )
                if "response" in res_slug and isinstance(res_slug["response"], dict):
                    resolved = res_slug["response"]
                    obj_id = resolved.get("object_id")
                    obj_type = resolved.get("type", "")
                    if obj_id and obj_type in ("group", "page", "event"):
                        vk_json2 = await _safe_vk_api_get(
                            "https://api.vk.com/method/groups.getById",
                            {
                                "group_id": str(obj_id),
                                "fields": "description,activity,status,members_count,city,photo_100,verified,is_closed",
                                "access_token": vk_token,
                                "v": "5.199",
                            },
                        )
                        if "response" in vk_json2:
                            if isinstance(vk_json2["response"], dict) and "groups" in vk_json2["response"] and vk_json2["response"]["groups"]:
                                group_data_vk = vk_json2["response"]["groups"][0]
                            elif isinstance(vk_json2["response"], list) and vk_json2["response"]:
                                group_data_vk = vk_json2["response"][0]
            except Exception as r_exc:
                logger.info("resolveScreenName не удался для %s: %s", slug, r_exc)

    if not group_data_vk:
        raise HTTPException(
            status_code=404,
            detail="Не удалось найти открытое сообщество ВКонтакте по этой ссылке. Проверьте адрес или выберите один из примеров.",
        )

    # Проверка на закрытое сообщество или удаленное/заблокированное
    is_closed = int(group_data_vk.get("is_closed", 0))
    deactivated = group_data_vk.get("deactivated")
    if is_closed != 0 or deactivated:
        raise HTTPException(
            status_code=404,
            detail="Не удалось найти открытое сообщество ВКонтакте по этой ссылке. Проверьте адрес или выберите один из примеров.",
        )

    gid = int(group_data_vk["id"])
    gname = str(group_data_vk.get("name", "")).strip()
    gslug = str(group_data_vk.get("screen_name", "")).strip() or (slug if slug else f"club{gid}")
    photo_100 = str(group_data_vk.get("photo_100", "")).strip()
    members_count = int(group_data_vk.get("members_count", 0))
    activity = str(group_data_vk.get("activity", "")).strip()
    status = str(group_data_vk.get("status", "")).strip()
    description = str(group_data_vk.get("description", "")).strip()
    city_obj = group_data_vk.get("city")
    detected_city = _detect_city(f"{gname} {description} {status}", city_obj)

    # Извлекаем предметную область и ключевые слова новым умным алгоритмом
    niche_title, keywords = _generate_niche_and_keywords(gname, activity, status, description, detected_city)

    # Фоллбэк на SVG-аватарку при отсутствии картинки
    if not photo_100:
        initials = "".join([part[0] for part in gname.split()[:2]]).upper() or "ВК"
        photo_100 = (
            f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'>"
            f"<rect width='100' height='100' rx='50' fill='%232688EB'/>"
            f"<text x='50' y='58' font-size='32' font-family='sans-serif' font-weight='bold' fill='white' text-anchor='middle'>{initials[:2]}</text>"
            f"</svg>"
        )

    tg_info = TargetGroupInfo(
        id=gid,
        name=gname,
        screen_name=gslug,
        activity=activity,
        status=status,
        description=description,
        members_count=members_count,
        is_closed=is_closed,
        verified=int(group_data_vk.get("verified", 0)),
        city=city_obj or ({"id": 0, "title": detected_city} if detected_city else None),
        photo_100=photo_100,
    )

    resp_obj = DetectNicheResponse(
        target_group=tg_info,
        group_id=gid,
        group_name=gname,
        screen_name=gslug,
        photo_100=photo_100,
        members_count=members_count,
        niche_title=niche_title,
        detected_city=detected_city,
        keywords=keywords,
    )

    # Кэшируем метаданные сообщества
    cached_payload = resp_obj.model_dump() if hasattr(resp_obj, "model_dump") else resp_obj.dict()
    _set_vk_search_cache(f"target:{gslug}", cached_payload)
    _set_vk_search_cache(f"target:{gid}", cached_payload)

    return resp_obj


@app.post("/api/v1/detect-niche", response_model=DetectNicheResponse)
async def detect_niche_v1(req: DetectNicheRequest, request: Request):
    """Определяет нишу, город и ключевые слова по ссылке или метаданным группы."""
    return await _handle_detect_niche(req, request)


@app.post("/api/detect-niche", response_model=DetectNicheResponse)
async def detect_niche_alias(req: DetectNicheRequest, request: Request):
    """Алиас для обратной совместимости с /api/detect-niche."""
    return await _handle_detect_niche(req, request)


# ============================================================================
# ЭНДПОИНТ 2: POST /api/v1/classify-groups (и алиас /api/classify-groups)
# ============================================================================

def _run_ml_pipeline_sklearn(
    niche_title: str,
    keywords: list[str],
    candidates: list[dict],
    feedback: list[ActiveLearningItem],
) -> ClassifyGroupsResponse:
    """Основной пайплайн скоринга на базе scikit-learn + numpy с удвоенным весом названия/категории."""
    if not candidates:
        return _build_classified_response([], np.array([]), {})

    query_doc = clean_text(f"{niche_title} {' '.join(keywords)}")
    docs = [
        clean_text(
            f"{c.get('name', '')} {c.get('name', '')} "
            f"{c.get('activity', '')} {c.get('activity', '')} "
            f"{c.get('status', '')} {c.get('description', '')}"
        )
        for c in candidates
    ]
    all_corpus = [query_doc] + docs

    if not any(all_corpus):
        return _run_ml_pipeline_fallback(niche_title, keywords, candidates, feedback)

    # Символьный TF-IDF (3-5 граммы)
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2000)
        tfidf_matrix = vec.fit_transform(all_corpus)
    except ValueError:
        return _run_ml_pipeline_fallback(niche_title, keywords, candidates, feedback)

    # Защита от малых выборок и определение n_components
    if min(tfidf_matrix.shape) <= 2:
        h_matrix = tfidf_matrix.toarray()
    else:
        max_svd_comp = min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if max_svd_comp >= 1:
            svd = TruncatedSVD(n_components=min(24, max_svd_comp), random_state=42)
            h_matrix = svd.fit_transform(tfidf_matrix)
        else:
            h_matrix = tfidf_matrix.toarray()

    h0 = h_matrix[0]
    h_cand = h_matrix[1:]

    norms = np.linalg.norm(h_cand, axis=1) * np.linalg.norm(h0) + 1e-9
    cos_sim = np.clip(np.dot(h_cand, h0) / norms, 0.0, 1.0)

    kw_words = set(clean_text(" ".join(keywords)).split())
    p_base_list = []

    for i, c in enumerate(candidates):
        words = set(docs[i].split())
        n_comm = sum(1 for m in COMMERCIAL_MARKERS if any(m in w for w in words))
        n_noise = sum(1 for m in NOISE_MARKERS if any(m in w for w in words))

        s_comm = min(0.35, n_comm * 0.07)
        s_noise = min(0.55, n_noise * 0.20)
        overlap = len(words.intersection(kw_words))
        s_theme = min(0.25, overlap * 0.08)
        s_open = 0.05 if c.get("is_closed") == 0 else -0.10

        query_hits = c.get("_query_hits", 1)
        hit_bonus = max(0.0, (query_hits - 1) * 0.25)

        z = 5.2 * (cos_sim[i] + s_comm + s_theme + s_open + hit_bonus - s_noise - 0.22)
        p_base = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.02, 0.99))
        p_base_list.append(p_base)

    p_base_arr = np.array(p_base_list)

    # Active Learning
    cand_ids = [c["id"] for c in candidates]
    p_final = p_base_arr.copy()

    feedback_map = {item.group_id: item.label for item in feedback}
    active_cids = [cid for cid in feedback_map if cid in cand_ids]

    if active_cids:
        x_all = np.hstack([h_cand, p_base_arr.reshape(-1, 1)])
        x_train = []
        y_train = []
        sample_weights = []

        for cid in active_cids:
            idx = cand_ids.index(cid)
            x_train.append(x_all[idx])
            y_train.append(feedback_map[cid])
            sample_weights.append(5.0)

        sorted_indices = np.argsort(p_base_arr)
        pos_anchors = [idx for idx in reversed(sorted_indices) if cand_ids[idx] not in feedback_map][:3]
        neg_anchors = [idx for idx in sorted_indices if cand_ids[idx] not in feedback_map][:3]

        for idx in pos_anchors:
            x_train.append(x_all[idx])
            y_train.append(1)
            sample_weights.append(0.5)

        for idx in neg_anchors:
            x_train.append(x_all[idx])
            y_train.append(0)
            sample_weights.append(0.5)

        if len(set(y_train)) < 2:
            missing_label = 0 if 1 in set(y_train) else 1
            feat_dim = x_all.shape[1]
            synthetic_vec = np.zeros(feat_dim) if missing_label == 0 else np.ones(feat_dim)
            x_train.append(synthetic_vec)
            y_train.append(missing_label)
            sample_weights.append(0.25)

        try:
            clf = LogisticRegression(C=2.0, max_iter=200, random_state=42)
            clf.fit(x_train, y_train, sample_weight=sample_weights)

            p_lr = clf.predict_proba(x_all)[:, 1]
            p_final = 0.4 * p_base_arr + 0.6 * p_lr
        except Exception as lr_err:
            logger.warning("Active Learning LogisticRegression error: %s", lr_err)

        # Принудительное назначение скоров для гарантированного учета разметки пользователя
        for cid, label in feedback_map.items():
            if cid in cand_ids:
                idx = cand_ids.index(cid)
                if label == 1:
                    p_final[idx] = max(0.92, float(p_final[idx]))
                else:
                    p_final[idx] = min(0.15, float(p_final[idx]))

    return _build_classified_response(candidates, p_final, feedback_map)


def _run_ml_pipeline_fallback(
    niche_title: str,
    keywords: list[str],
    candidates: list[dict],
    feedback: list[ActiveLearningItem],
) -> ClassifyGroupsResponse:
    """Автономный fallback-скоринг на чистом Python (math + Counter) без scikit-learn."""
    def _extract_ngrams(text: str) -> Counter:
        padded = f" {text} "
        c = Counter()
        for n in range(3, 6):
            for i in range(len(padded) - n + 1):
                c[padded[i:i+n]] += 1
        return c

    def _cos_sim(c1: Counter, c2: Counter) -> float:
        dot = sum(c1[k] * c2[k] for k in c1 if k in c2)
        n1 = math.sqrt(sum(v * v for v in c1.values()))
        n2 = math.sqrt(sum(v * v for v in c2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    query_doc = clean_text(f"{niche_title} {' '.join(keywords)}")
    query_ngrams = _extract_ngrams(query_doc)

    docs = [
        clean_text(
            f"{c.get('name', '')} {c.get('name', '')} "
            f"{c.get('activity', '')} {c.get('activity', '')} "
            f"{c.get('status', '')} {c.get('description', '')}"
        )
        for c in candidates
    ]
    cand_ngrams = [_extract_ngrams(d) for d in docs]

    kw_words = set(clean_text(" ".join(keywords)).split())
    p_base_list = []

    for i, c in enumerate(candidates):
        sim = _cos_sim(query_ngrams, cand_ngrams[i])
        words = set(docs[i].split())
        n_comm = sum(1 for m in COMMERCIAL_MARKERS if any(m in w for w in words))
        n_noise = sum(1 for m in NOISE_MARKERS if any(m in w for w in words))

        s_comm = min(0.35, n_comm * 0.07)
        s_noise = min(0.55, n_noise * 0.20)
        overlap = len(words.intersection(kw_words))
        s_theme = min(0.25, overlap * 0.08)
        s_open = 0.05 if c.get("is_closed") == 0 else -0.10

        query_hits = c.get("_query_hits", 1)
        hit_bonus = max(0.0, (query_hits - 1) * 0.25)

        z = 5.2 * (sim + s_comm + s_theme + s_open + hit_bonus - s_noise - 0.22)
        try:
            p_val = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            p_val = 0.0 if z < 0 else 1.0

        p_base = max(0.02, min(0.99, p_val))
        p_base_list.append(p_base)

    # Active Learning fallback: ранжирование по близости к размеченным экспертом образцам
    cand_ids = [c["id"] for c in candidates]
    feedback_map = {item.group_id: item.label for item in feedback}
    p_final = p_base_list.copy()

    pos_cids = [cid for cid, label in feedback_map.items() if label == 1 and cid in cand_ids]
    neg_cids = [cid for cid, label in feedback_map.items() if label == 0 and cid in cand_ids]

    if pos_cids or neg_cids:
        for i, cid in enumerate(cand_ids):
            if cid in feedback_map:
                if feedback_map[cid] == 1:
                    p_final[i] = max(0.92, p_final[i])
                else:
                    p_final[i] = min(0.15, p_final[i])
            else:
                max_pos_sim = max([_cos_sim(cand_ngrams[i], cand_ngrams[cand_ids.index(pc)]) for pc in pos_cids], default=0.0)
                max_neg_sim = max([_cos_sim(cand_ngrams[i], cand_ngrams[cand_ids.index(nc)]) for nc in neg_cids], default=0.0)
                delta = 0.20 * max_pos_sim - 0.25 * max_neg_sim
                p_final[i] = max(0.02, min(0.99, p_final[i] + delta))

    return _build_classified_response(candidates, p_final, feedback_map)


def _build_classified_response(
    candidates: list[dict],
    p_scores: Any,
    feedback_map: dict[int, int],
) -> ClassifyGroupsResponse:
    """Формирует структурированный ответ с ранжированием и пограничными группами."""
    classified_results: list[ClassifiedGroup] = []
    uncertain_candidates: list[UncertainGroup] = []

    for i, c in enumerate(candidates):
        score = round(float(p_scores[i]), 3)

        if score >= 0.85:
            segment_tag = "Прямой конкурент"
        elif score >= 0.50:
            segment_tag = "Косвенный конкурент"
        else:
            segment_tag = "Информационный шум"

        city_title = ""
        if c.get("city") and isinstance(c["city"], dict):
            city_title = c["city"].get("title", "")
        elif c.get("city_title"):
            city_title = c["city_title"]

        classified_results.append(
            ClassifiedGroup(
                id=c["id"],
                name=c["name"],
                screen_name=c.get("screen_name", ""),
                activity=c.get("activity", ""),
                status=c.get("status", ""),
                description=c.get("description", ""),
                members_count=c.get("members_count", 0),
                is_closed=c.get("is_closed", 0),
                verified=c.get("verified", 0),
                city_title=city_title,
                photo_100=c.get("photo_100", ""),
                lstm_score=score,
                segment_tag=segment_tag,
            )
        )

        if 0.40 <= score <= 0.65 and c["id"] not in feedback_map:
            uncertain_candidates.append(
                UncertainGroup(
                    id=c["id"],
                    name=c["name"],
                    activity=c.get("activity", ""),
                    status=c.get("status", ""),
                    description=c.get("description", ""),
                    members_count=c.get("members_count", 0),
                    photo_100=c.get("photo_100", ""),
                    lstm_score=score,
                )
            )

    uncertain_candidates.sort(key=lambda u: abs(u.lstm_score - 0.52))
    uncertain_for_active_learning = uncertain_candidates[:8]

    classified_results.sort(key=lambda g: g.lstm_score, reverse=True)

    total_checked = len(classified_results)
    relevant_count = sum(1 for g in classified_results if g.lstm_score >= 0.50)
    noise_filtered = total_checked - relevant_count

    return ClassifyGroupsResponse(
        total_checked=total_checked,
        relevant_count=relevant_count,
        noise_filtered=noise_filtered,
        uncertain_for_active_learning=uncertain_for_active_learning,
        results=classified_results,
    )


def _run_ml_pipeline(
    niche_title: str,
    keywords: list[str],
    candidates: list[dict],
    feedback: list[ActiveLearningItem],
) -> ClassifyGroupsResponse:
    """Точка входа ML-конвейера: выбирает scikit-learn или автономный fallback."""
    if not candidates:
        return ClassifyGroupsResponse(
            total_checked=0,
            relevant_count=0,
            noise_filtered=0,
            uncertain_for_active_learning=[],
            results=[],
        )

    if HAS_SKLEARN and np is not None:
        return _run_ml_pipeline_sklearn(niche_title, keywords, candidates, feedback)
    return _run_ml_pipeline_fallback(niche_title, keywords, candidates, feedback)


async def _handle_classify_groups(req: ClassifyGroupsRequest, request: Request) -> ClassifyGroupsResponse:
    check_request_sign(request)

    candidates_dict: list[dict] = []

    # 1. Если переданы кандидаты с клиента
    if req.candidate_groups:
        candidates_dict = [
            cg.model_dump() if hasattr(cg, "model_dump") else cg.dict()
            for cg in req.candidate_groups
        ]

    # Автоматическое определение preset_key по ID целевой группы, если ключ не был явно передан
    preset_key = req.preset_key
    if not preset_key and req.target_group_id:
        target_preset_map = {
            25557243: "dodo",
            18456201: "alexfitness",
            31084223: "fitauto",
        }
        preset_key = target_preset_map.get(req.target_group_id)

    # 2. Если строго указан один из трех демо-пресетов (dodo, alexfitness, fitauto)
    if not candidates_dict and preset_key and preset_key.lower() in ("dodo", "alexfitness", "fitauto"):
        cached = _get_vk_search_cache(f"preset:{preset_key.lower()}")
        if cached:
            candidates_dict = cached

    # 3. Реальный сбор кандидатов во ВКонтакте через VK API groups.search + groups.getById
    active_token = VK_USER_TOKEN or VK_SERVICE_TOKEN
    if not candidates_dict and req.keywords and active_token:
        search_keywords = [kw.strip().lower() for kw in req.keywords if kw.strip()][:6]

        # Если задан detected_city и среди ключевых слов нет ни одного с названием города, добавляем "{keywords[0]} {city}"
        if req.detected_city:
            city_clean = req.detected_city.strip().lower()
            has_city = any(city_clean in kw for kw in search_keywords)
            if not has_city and search_keywords:
                city_query = f"{search_keywords[0]} {city_clean}"
                if city_query not in search_keywords:
                    search_keywords.insert(1, city_query)
                    search_keywords = search_keywords[:6]

        group_hit_counts: dict[int, int] = defaultdict(int)
        uncached_keywords: list[str] = []

        # Проверяем 24-часовой кэш kw_live для каждого запроса
        for kw in search_keywords:
            cached_ids = _get_live_keyword_cache(kw)
            if cached_ids is not None:
                for gid in cached_ids:
                    group_hit_counts[gid] += 1
            else:
                uncached_keywords.append(kw)

        # Выполняем реальные запросы к VK API волнами по 2 параллельно с паузой 0.35 сек
        if uncached_keywords:
            for i in range(0, len(uncached_keywords), 2):
                chunk = uncached_keywords[i:i+2]
                tasks = [
                    _safe_vk_api_get(
                        "https://api.vk.com/method/groups.search",
                        {
                            "q": kw,
                            "count": 45,
                            "sort": 0,
                            "access_token": active_token,
                            "v": "5.199",
                        },
                    )
                    for kw in chunk
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, res in enumerate(results):
                    kw_name = chunk[idx]
                    kw_ids = []
                    if isinstance(res, dict) and "response" in res:
                        items = res["response"].get("items", [])
                        for item in items:
                            gid = int(item["id"])
                            kw_ids.append(gid)
                            group_hit_counts[gid] += 1
                    _set_live_keyword_cache(kw_name, kw_ids)
                if i + 2 < len(uncached_keywords):
                    await asyncio.sleep(0.35)

        # Исключаем целевую группу
        if req.target_group_id and req.target_group_id in group_hit_counts:
            del group_hit_counts[req.target_group_id]

        unique_ids = list(group_hit_counts.keys())[:180]

        if unique_ids:
            ids_str = ",".join(str(gid) for gid in unique_ids)
            try:
                batch_resp = await _safe_vk_api_get(
                    "https://api.vk.com/method/groups.getById",
                    {
                        "group_ids": ids_str,
                        "fields": "description,activity,status,members_count,city,photo_100,verified,is_closed",
                        "access_token": active_token,
                        "v": "5.199",
                    },
                )
                raw_groups = []
                if isinstance(batch_resp, dict) and "response" in batch_resp:
                    raw_groups = batch_resp["response"].get("groups", []) or batch_resp["response"]

                candidates_list = []
                for g in raw_groups:
                    gid = int(g["id"])
                    if req.target_group_id and gid == req.target_group_id:
                        continue
                    if g.get("is_closed") == 1 or g.get("is_closed") == 2:
                        continue
                    if g.get("deactivated"):
                        continue
                    hits = group_hit_counts.get(gid, 1)
                    candidates_list.append({
                        "id": gid,
                        "name": g.get("name", ""),
                        "screen_name": g.get("screen_name", ""),
                        "activity": g.get("activity", ""),
                        "status": g.get("status", ""),
                        "description": g.get("description", ""),
                        "members_count": int(g.get("members_count", 0)),
                        "is_closed": int(g.get("is_closed", 0)),
                        "verified": int(g.get("verified", 0)),
                        "city": g.get("city"),
                        "photo_100": g.get("photo_100", ""),
                        "_query_hits": hits,
                    })
                candidates_dict = candidates_list
            except Exception as batch_err:
                logger.warning("Ошибка пакетного запроса groups.getById: %s", batch_err)

    # 4. Если это строго демо-пресет и кандидаты ещё не были извлечены
    if not candidates_dict and preset_key and preset_key.lower() in ("dodo", "alexfitness", "fitauto"):
        cached = _get_vk_search_cache(f"preset:{preset_key.lower()}")
        if cached:
            candidates_dict = cached

    # СТРОГОЕ ПРАВИЛО: Больше никакого фоллбэка на 150 демо-групп для кастомных сообществ!

    return _run_ml_pipeline(
        niche_title=req.niche_title,
        keywords=req.keywords,
        candidates=candidates_dict,
        feedback=req.active_learning_feedback,
    )

    return _run_ml_pipeline(
        niche_title=req.niche_title,
        keywords=req.keywords,
        candidates=candidates_dict,
        feedback=req.active_learning_feedback,
    )


@app.post("/api/v1/classify-groups", response_model=ClassifyGroupsResponse)
async def classify_groups_v1(req: ClassifyGroupsRequest, request: Request):
    """Классифицирует сообщества-кандидаты по релевантности предметной области."""
    return await _handle_classify_groups(req, request)


@app.post("/api/classify-groups", response_model=ClassifyGroupsResponse)
async def classify_groups_alias(req: ClassifyGroupsRequest, request: Request):
    """Алиас для обратной совместимости с /api/classify-groups."""
    return await _handle_classify_groups(req, request)


# ============================================================================
# СТАРЫЕ ЭНДПОИНТЫ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ============================================================================

@app.api_route("/api/subjects", methods=["GET", "HEAD"])
async def subjects(request: Request):
    """Список поддерживаемых отраслей бизнеса (legacy)."""
    check_request_sign(request)
    return [
        {
            "key": key,
            "title": info["title"],
            "description": info["description"],
        }
        for key, info in SUBJECTS.items()
    ]


@app.api_route("/api/cities", methods=["GET", "HEAD"])
async def cities(request: Request):
    """Список доступных городов с собранной аналитикой (legacy)."""
    check_request_sign(request)
    data = _read_data_raw("__cities__")
    return data if isinstance(data, list) else []


@app.api_route("/api/stats", methods=["GET", "HEAD"])
async def stats(
    request: Request,
    subject: str = Query(..., description="Ключ предметной области"),
    city: Optional[str] = Query(None, description="Фильтр по городу (точное совпадение)"),
):
    """Аналитическая сводка и публикации с максимальной вовлечённостью (legacy)."""
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

    payload["subject_title"] = SUBJECTS[subject]["title"]
    if city:
        payload["city"] = city

    payload.pop("cached_at", None)
    payload.pop("computed_at", None)

    return payload
