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
from collections import Counter
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

# Защищённый ключ приложения VK (ID 54674554) и сервисные токены
VK_SECRET_KEY = os.environ.get("VK_SECRET_KEY") or os.environ.get("VK_SECURE_KEY") or ""
VK_SIGN_REQUIRED = os.environ.get("VK_SIGN_REQUIRED", "true").lower() in ("true", "1", "yes")
VK_SERVICE_TOKEN = os.environ.get("VK_SERVICE_TOKEN") or ""
VK_USER_TOKEN = os.environ.get("VK_USER_TOKEN") or ""

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
    "вконтакте", "паблик", "подпишись", "добро", "пожаловать", "россия",
}

COMMERCIAL_MARKERS = {
    "доставка", "заказать", "заказ", "цена", "цены", "прайс", "стоимость", "купить",
    "магазин", "услуги", "услуга", "запись", "записаться", "каталог", "меню", "работаем",
    "адрес", "акция", "акции", "скидка", "скидки", "производство", "сервис", "студия",
    "клиника", "клуб", "тренировка", "тренировки", "абонемент", "ремонт", "сто",
    "шиномонтаж", "диагностика", "пицца", "роллы", "суши", "кафе", "ресторан",
    "бар", "салон", "выпечка", "сеты",
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
        "*",
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

def clean_text(text: str) -> str:
    """Очищает текст от ссылок, телефонов, эмодзи и спецсимволов."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+", " ", text.lower())
    text = re.sub(r"\[id\d+\|[^\]]+\]", " ", text)
    text = re.sub(r"\[club\d+\|[^\]]+\]", " ", text)
    text = re.sub(r"@[a-z0-9_]+", " ", text)
    text = re.sub(r"\+?[0-9\-\(\)\s]{7,18}", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    words = [w for w in text.split() if len(w) > 1 and w not in RUSSIAN_STOP_WORDS]
    return " ".join(words)


def extract_slug(input_str: str) -> str:
    """Извлекает короткое имя сообщества из ссылки или текстового ввода."""
    if not input_str:
        return ""
    s = input_str.strip().rstrip("/")
    if "vk.com/" in s:
        s = s.split("vk.com/")[-1]
    if s.startswith("@"):
        s = s[1:]
    return s.split("?")[0].strip()


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
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url, params=params)
            return resp.json()

    def _sync_fetch() -> dict:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
        req = urllib.request.Request(full_url, headers={"User-Agent": "NeurokontRadar/1.0"})
        with urllib.request.urlopen(req, timeout=4.0) as r:
            return json.loads(r.read().decode("utf-8"))

    return await asyncio.to_thread(_sync_fetch)


def _generate_niche_and_keywords(
    name: str,
    activity: str,
    status: str,
    description: str,
    detected_city: str,
) -> tuple[str, list[str]]:
    """Извлекает предметную область и 5–8 ключевых запросов с помощью TF-IDF или Counter fallback."""
    clean_name = clean_text(name.split("|")[0].split("-")[0].split("/")[0])
    clean_act = clean_text(activity)

    if clean_act and len(clean_act) > 3 and clean_act not in ("блог", "публичная страница"):
        niche_title = f"{activity} • {clean_name.title()}" if clean_name else activity
    else:
        niche_title = clean_name.title() or "Локальный бизнес и услуги"

    full_corpus = clean_text(f"{name} {activity} {status} {description}")

    # Удаляем название города из ключевых слов, чтобы не сужать поиск
    city_word = clean_text(detected_city).lower()
    if city_word:
        full_corpus = " ".join([w for w in full_corpus.split() if w != city_word])

    extracted_keywords: list[str] = []

    if clean_act and len(clean_act) > 3 and clean_act not in ("блог", "публичная страница"):
        extracted_keywords.append(activity.lower())

    if clean_name and len(clean_name) > 3:
        extracted_keywords.append(clean_name.lower())

    # Экстракция униграмм и биграмм: TF-IDF (scikit-learn) или Counter (чистый Python)
    corpus_words = full_corpus.split()
    if len(corpus_words) >= 2:
        if HAS_SKLEARN:
            try:
                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=25,
                    stop_words=list(RUSSIAN_STOP_WORDS),
                )
                vec.fit([full_corpus])
                terms = vec.get_feature_names_out()
                for t in terms:
                    t_str = t.strip()
                    if len(t_str) > 3 and t_str not in extracted_keywords and not t_str.isdigit():
                        extracted_keywords.append(t_str)
                    if len(extracted_keywords) >= 8:
                        break
            except Exception:
                pass
        else:
            # Fallback на Counter n-грамм
            words = [w for w in corpus_words if len(w) > 3 and w not in RUSSIAN_STOP_WORDS]
            word_counts = Counter(words)
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            bg_counts = Counter(bigrams)

            for bg, _ in bg_counts.most_common(4):
                if bg not in extracted_keywords:
                    extracted_keywords.append(bg)
            for w, _ in word_counts.most_common(6):
                if w not in extracted_keywords:
                    extracted_keywords.append(w)
                if len(extracted_keywords) >= 8:
                    break

    # Фоллбэк по умолчанию, если ключевых фраз мало
    default_candidates = [
        "доставка еды", "заказать онлайн", "пиццерия", "суши роллы",
        "фитнес клуб", "тренажерный зал", "автосервис", "ремонт авто",
    ]
    for dc in default_candidates:
        if len(extracted_keywords) >= 6:
            break
        if dc not in extracted_keywords:
            extracted_keywords.append(dc)

    return niche_title, extracted_keywords[:8]


# ============================================================================
# Pydantic-схемы (v2)
# ============================================================================

class TargetGroupInfo(BaseModel):
    id: int
    name: str
    screen_name: Optional[str] = ""
    activity: Optional[str] = ""
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
    description: Optional[str] = ""
    members_count: int = 0
    photo_100: Optional[str] = ""
    lstm_score: float


class ClassifiedGroup(BaseModel):
    id: int
    name: str
    screen_name: Optional[str] = ""
    activity: Optional[str] = ""
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

    # Проверка на сторонний домен не из ВК
    if raw_input.startswith("http://") or raw_input.startswith("https://"):
        if "vk.com" not in raw_input and "vk.ru" not in raw_input:
            raise HTTPException(
                status_code=404,
                detail="Не удалось распознать открытое сообщество ВКонтакте. Проверьте ссылку или выберите один из демо-примеров ниже.",
            )

    slug = extract_slug(raw_input).lower()

    if slug in ("closed_test_error", "error_closed", "forbidden"):
        raise HTTPException(
            status_code=404,
            detail="Сообщество закрыто или недоступно для анализа. Укажите открытое сообщество или выберите демо-пример.",
        )

    # 1. Проверяем кэш пресетов (detect:slug или preset_meta:slug)
    if slug:
        cached_meta = _get_vk_search_cache(f"detect:{slug}") or _get_vk_search_cache(f"preset_meta:{slug}")
        if cached_meta:
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

    # 2. Если передан токен ВК — пробуем получить реальные данные группы через VK API
    vk_token = VK_USER_TOKEN or VK_SERVICE_TOKEN
    group_data_vk: Optional[dict] = None

    if vk_token and (slug or req.group_id):
        try:
            group_ident = str(req.group_id) if req.group_id else slug
            vk_json = await _safe_vk_api_get(
                "https://api.vk.com/method/groups.getById",
                {
                    "group_id": group_ident,
                    "fields": "description,activity,status,members_count,city,verified,is_closed",
                    "access_token": vk_token,
                    "v": "5.199",
                },
            )
            if "response" in vk_json and "groups" in vk_json["response"] and vk_json["response"]["groups"]:
                group_data_vk = vk_json["response"]["groups"][0]
            elif "response" in vk_json and isinstance(vk_json["response"], list) and vk_json["response"]:
                group_data_vk = vk_json["response"][0]
        except Exception as exc:
            logger.info("Запрос к VK API groups.getById не удался: %s", exc)

    # 3. Формируем профиль группы
    if group_data_vk:
        gid = int(group_data_vk.get("id", 10001))
        gname = group_data_vk.get("name", "Сообщество ВКонтакте")
        gslug = group_data_vk.get("screen_name", slug)
        gact = group_data_vk.get("activity", "")
        gdesc = group_data_vk.get("description", "")
        gstat = group_data_vk.get("status", "")
        gmem = int(group_data_vk.get("members_count", 0))
        gclosed = int(group_data_vk.get("is_closed", 0))
        gver = int(group_data_vk.get("verified", 0))
        gcity = group_data_vk.get("city")
        gphoto = group_data_vk.get("photo_100", "")
    elif req.name or req.description:
        gid = req.group_id or 10001
        gname = req.name or (slug.replace("_", " ").title() if slug else "Сообщество ВКонтакте")
        gslug = slug or "community"
        gact = req.activity or "Бизнес и услуги"
        gdesc = req.description or ""
        gstat = req.status or ""
        gmem = 15400
        gclosed = 0
        gver = 0
        gcity = {"id": 110, "title": req.city} if req.city else None
        gphoto = ""
    elif slug:
        # Синтез валидного профиля по slug
        gid = 10001 + abs(hash(slug)) % 90000
        gname = slug.replace("_", " ").replace(".", " ").title()
        gslug = slug
        gact = "Бизнес и услуги"
        gdesc = f"Официальная страница {gname} во ВКонтакте. Товары, услуги, специальные предложения и клиентский сервис."
        gstat = "Работаем ежедневно. Консультации и онлайн-заказ."
        gmem = 24500
        gclosed = 0
        gver = 0
        gcity = {"id": 110, "title": "Пермь"}
        gphoto = ""
    else:
        raise HTTPException(
            status_code=404,
            detail="Не удалось распознать открытое сообщество ВКонтакте. Проверьте ссылку или выберите один из демо-примеров ниже.",
        )

    # Генерация аккуратного SVG аватара при отсутствии URL фото
    if not gphoto:
        initials = "".join([part[0] for part in gname.split()[:2]]).upper() or "ВК"
        gphoto = (
            f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'>"
            f"<rect width='100' height='100' rx='50' fill='%232688EB'/>"
            f"<text x='50' y='58' font-size='32' font-family='sans-serif' font-weight='bold' fill='white' text-anchor='middle'>{initials[:2]}</text>"
            f"</svg>"
        )

    detected_city = _detect_city(f"{gname} {gdesc} {gstat}", gcity)
    niche_title, keywords = _generate_niche_and_keywords(gname, gact, gstat, gdesc, detected_city)

    tg_info = TargetGroupInfo(
        id=gid,
        name=gname,
        screen_name=gslug,
        activity=gact,
        description=gdesc,
        members_count=gmem,
        is_closed=gclosed,
        verified=gver,
        city=gcity or {"id": 110, "title": detected_city},
        photo_100=gphoto,
    )

    return DetectNicheResponse(
        target_group=tg_info,
        group_id=gid,
        group_name=gname,
        screen_name=gslug,
        photo_100=gphoto,
        members_count=gmem,
        niche_title=niche_title,
        detected_city=detected_city,
        keywords=keywords,
    )


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
    """Основной пайплайн скоринга на базе scikit-learn + numpy."""
    query_doc = clean_text(f"{niche_title} {' '.join(keywords)}")
    docs = [
        clean_text(f"{c.get('name', '')} {c.get('activity', '')} {c.get('description', '')}")
        for c in candidates
    ]

    # Символьный TF-IDF (3-5 граммы)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2000)
    all_corpus = [query_doc] + docs
    tfidf_matrix = vec.fit_transform(all_corpus)

    # Сжатие в скрытые представления TruncatedSVD
    n_comp = min(24, max(2, len(docs) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    h_matrix = svd.fit_transform(tfidf_matrix)

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

        z = 5.2 * (cos_sim[i] + s_comm + s_theme + s_open - s_noise - 0.22)
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

        try:
            clf = LogisticRegression(C=2.0, max_iter=200, random_state=42)
            clf.fit(x_train, y_train, sample_weight=sample_weights)

            p_lr = clf.predict_proba(x_all)[:, 1]
            p_final = 0.4 * p_base_arr + 0.6 * p_lr

            for cid, label in feedback_map.items():
                if cid in cand_ids:
                    idx = cand_ids.index(cid)
                    if label == 1:
                        p_final[idx] = max(0.92, p_final[idx])
                    else:
                        p_final[idx] = min(0.15, p_final[idx])
        except Exception as lr_err:
            logger.warning("Active Learning LogisticRegression error: %s", lr_err)

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
        clean_text(f"{c.get('name', '')} {c.get('activity', '')} {c.get('description', '')}")
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

        z = 5.2 * (sim + s_comm + s_theme + s_open - s_noise - 0.22)
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
        candidates_dict = [cg.model_dump() for cg in req.candidate_groups]

    # 2. Если указан пресет (dodo, alexfitness, fitauto)
    if not candidates_dict and req.preset_key:
        cached = _get_vk_search_cache(f"preset:{req.preset_key.lower()}")
        if cached:
            candidates_dict = cached

    # 3. Поиск по ключевым словам в локальном кэше
    if not candidates_dict and req.keywords:
        seen_ids = set()
        merged = []
        for kw in req.keywords:
            kw_clean = kw.strip().lower()
            cached_kw = _get_vk_search_cache(f"kw:{kw_clean}")
            if cached_kw and isinstance(cached_kw, list):
                for item in cached_kw:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        merged.append(item)
        if merged:
            candidates_dict = merged

    # 4. Если задан пользовательский токен VK и кэша нет — ищем через VK API
    if not candidates_dict and VK_USER_TOKEN and req.keywords:
        try:
            seen_ids = set()
            fetched_groups = []
            for q in req.keywords[:3]:
                s_json = await _safe_vk_api_get(
                    "https://api.vk.com/method/groups.search",
                    {
                        "q": q,
                        "type": "group,page",
                        "count": 25,
                        "access_token": VK_USER_TOKEN,
                        "v": "5.199",
                    },
                )
                items = s_json.get("response", {}).get("items", [])
                new_ids = [str(it["id"]) for it in items if it["id"] not in seen_ids]
                if new_ids:
                    seen_ids.update([int(i) for i in new_ids])
                    g_json = await _safe_vk_api_get(
                        "https://api.vk.com/method/groups.getById",
                        {
                            "group_ids": ",".join(new_ids),
                            "fields": "description,activity,members_count,city,verified,is_closed",
                            "access_token": VK_USER_TOKEN,
                            "v": "5.199",
                        },
                    )
                    groups_info = g_json.get("response", {}).get("groups", []) or g_json.get("response", [])
                    for g in groups_info:
                        fetched_groups.append({
                            "id": g["id"],
                            "name": g.get("name", ""),
                            "screen_name": g.get("screen_name", ""),
                            "activity": g.get("activity", ""),
                            "description": g.get("description", ""),
                            "members_count": g.get("members_count", 0),
                            "is_closed": g.get("is_closed", 0),
                            "verified": g.get("verified", 0),
                            "city": g.get("city"),
                            "photo_100": g.get("photo_100", ""),
                        })
            if fetched_groups:
                candidates_dict = fetched_groups
                if req.keywords:
                    _set_vk_search_cache(f"kw:{req.keywords[0].strip().lower()}", fetched_groups)
        except Exception as vk_err:
            logger.warning("Ошибка при вызове VK API groups.search: %s", vk_err)

    # 5. Фоллбэк: если кастомный ввод без токена — собираем кандидатов из локальной базы
    if not candidates_dict:
        combined = []
        seen_ids = set()
        for p_key in ("dodo", "alexfitness", "fitauto"):
            p_data = _get_vk_search_cache(f"preset:{p_key}")
            if p_data and isinstance(p_data, list):
                for item in p_data:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        combined.append(item)
        candidates_dict = combined

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
