#!/usr/bin/env python3
"""
Stateful Telegram-интеграция для Neurotutor.

Поднимается над telegram-bot-agent: читает inbox, диспетчеризует по
ключевым словам, хранит состояние сессии в файлах
`data/tg_session_{user_id}.json`.

Запуск:
    cd /root/neurotutor && venv/bin/python tg_handler.py

Требования (должны быть в .env):
    TELEGRAM_BOT_TOKEN=<токен>
    TELEGRAM_USER_ID=<telegram user id>
"""

from __future__ import annotations

import functools
import html
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from neurotutor.agent import drill

# ── конфиг ──────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
SESSION_DIR = DATA_DIR / "tg_sessions"
INBOX       = Path("/tmp/telegram_inbox.json")

# как долго хранить историю (сообщений от пользователя)
MAX_HISTORY = 10

# ── логирование ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tg_handler")

# ── состояние сессии ────────────────────────────────────────────────────────


@dataclass
class SessionState:
    mode:      str = "new"      # diagnostic|review|case|osce|imaging|new
    persona:   str = "corvin"   # corvin|lin|plain
    step:      int = 0          # счётчик шагов (для многошаговых режимов)
    history:   list[dict] = field(default_factory=list)
    case_final: bool = False    # кейс переведён в режим финального ответа
    updated_at: str = ""


def session_path(user_id: int | str) -> Path:
    return SESSION_DIR / f"tg_session_{user_id}.json"


def load_session(user_id: int | str) -> SessionState:
    p = session_path(user_id)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # миграция: history могла быть списком строк
            hist = []
            for h in data.get("history", []):
                if isinstance(h, dict) and "role" in h:
                    hist.append(h)
            data["history"] = hist
            return SessionState(**{k: v for k, v in data.items()
                                   if k in SessionState.__dataclass_fields__})
        except Exception:
            log.exception("session file corrupted, starting fresh")
    return SessionState()


def save_session(user_id: int | str, state: SessionState) -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    p = session_path(user_id)
    p.write_text(json.dumps(
        {**state.__dict__, "history": state.history[-MAX_HISTORY:]},
        ensure_ascii=False, indent=2
    ))


def clear_session(user_id: int | str) -> None:
    p = session_path(user_id)
    if p.exists():
        p.unlink()


# ── маршрутизация по ключевым словам ────────────────────────────────────────

RESET_PHRASES = {"стоп", "выход", "новая тема", "новая сессия", "/reset", "/new"}
DIAGNOSTIC_KW = {"проверь меня", "диагностика", "тест", "проверка", "оцени меня",
                 "проверь", "найди пробелы", "выяви пробелы"}
REVIEW_KW     = {"повторение", "карточки", "повтори", "на повторение",
                 "что на сегодня", "что повторить"}
CASE_KW       = {"кейс", "клинический случай", "разбор", "case", "кейс"}
OSCE_KW        = {"оскэ", "экзамен", "станция", "осцэ", "/osce", "/exam"}
IMAGING_KW    = {"кт", "мрт", "снимок", "ангио", "imaging", "кт/мрт"}
NEW_KW        = {"объясни", "расскажи", "новое", "новый", "что такое", "давай учить"}
CORVIN_KW     = {"персонаж корвин", "корвин", "профессор", "/corvin"}
# NB: персонаж — «Линь». Раньше ключевик был «лин» (без «ь») и матчился как
# подстрока внутри «к-лин-ический», «лин-ия» и т.п. → ложный persona:lin.
LIN_KW        = {"персонаж линь", "линь", "май линь", "/lin"}


@functools.lru_cache(maxsize=1024)
def _kw_pattern(kw: str) -> re.Pattern:
    # Левая граница слова: совпадение должно НАЧИНАТЬСЯ на стыке слов, но может
    # продолжаться внутрь слова — так «повтори» всё ещё ловит «повторить», а
    # «лин»/«кт» больше НЕ цепляются изнутри «клинический»/«контакт».
    return re.compile(r"(?<!\w)" + re.escape(kw))


def _has_kw(t: str, kws: set[str]) -> bool:
    return any(_kw_pattern(kw).search(t) for kw in kws)


def route(text: str) -> tuple[str, str]:
    """Вернуть (mode, persona). text в lowercase."""
    t = text.lower().strip()

    # сначала сброс — он важнее всего
    if _has_kw(t, RESET_PHRASES):
        return "reset", ""

    # персонажи (не меняют mode)
    if _has_kw(t, CORVIN_KW):
        return "persona:corvin", "corvin"
    if _has_kw(t, LIN_KW):
        return "persona:lin", "lin"

    # режимы
    if _has_kw(t, DIAGNOSTIC_KW):
        return "diagnostic", ""
    if _has_kw(t, REVIEW_KW):
        return "review", ""
    if _has_kw(t, CASE_KW):
        return "case", ""
    if _has_kw(t, OSCE_KW):
        return "osce", ""
    if _has_kw(t, IMAGING_KW):
        return "imaging", ""
    if _has_kw(t, NEW_KW):
        return "new", ""

    return "", ""   # продолжить текущий mode (empty = keep)


def apply_route(state: SessionState, routing: str, persona_hint: str) -> SessionState:
    if routing == "reset":
        state = SessionState()
        return state
    if routing.startswith("persona:"):
        state.persona = persona_hint or routing.split(":", 1)[1]
        return state
    if routing:
        state.mode  = routing
        state.step  = 0
        state.history = []     # сброс истории при смене режима
        # persona не трогаем
    return state


# ──Neurotutor CLI wrapper ─────────────────────────────────────────────────────


def neurotutor_reply(mode: str, user_message: str,
                    persona: str, history: list[dict]) -> str:
    """Сгенерировать ответ агента в процессе (без subprocess/CLI).

    history — список dict {role, content} предыдущих ходов; передаётся
    оркестратору КАК ЕСТЬ, чтобы у диалога была настоящая память. Раньше
    история клеилась в текст и терялась — отсюда «ботовость» и ответы вида
    «я не храню контекст». Прямой вызов session.turn() также убирает протечку
    служебного префикса «role / persona:» из CLI-печати.
    """
    from neurotutor.session import (turn, start as start_session,
                                    end as end_session)

    hist = [h for h in (history or [])[-(MAX_HISTORY * 2):]
            if h.get("role") in ("user", "assistant") and h.get("content")]

    sid = None
    try:
        sid = start_session(mode, notes="tg chat")
    except Exception:
        log.exception("session start failed (continuing without persist)")
        sid = None
    try:
        out = turn(mode, user_message, history=hist,
                   persona=persona, session_id=sid)
        return (out.get("reply") or "").strip() or "[пустой ответ — переформулируй?]"
    except Exception:
        log.exception("neurotutor reply failed")
        return "[внутренняя ошибка, попробуй позже]"
    finally:
        if sid is not None:
            try:
                end_session(sid)
            except Exception:
                log.exception("session end failed")


# ── утреннее напоминание (due cards) ─────────────────────────────────────────


def due_cards_summary() -> str:
    """Вернуть строку с напоминанием о карточках или пустую."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "neurotutor.cli", "due"],
            capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ""
        lines = result.stdout.strip().splitlines()
        # берём только первые 5 карточек
        summary = "📚 <b>На повторение сегодня:</b>\n"
        shown = 0
        for line in lines[1:]:   # пропускаем заголовок таблицы
            if shown >= 5:
                summary += f"...и ещё {len(lines)-6} карточек. Начни /review"
                break
            summary += f"• {line.strip()}\n"
            shown += 1
        return summary.strip()
    except Exception:
        log.exception("due check failed")
        return ""


# ── Telegram отправка ────────────────────────────────────────────────────────


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER = os.getenv("TELEGRAM_USER_ID", "")

if TELEGRAM_USER:
    TELEGRAM_USER = str(TELEGRAM_USER).lstrip("@")


def esc(text: object) -> str:
    """Экранировать динамический (LLM/пользовательский) текст для HTML-режима.

    Telegram с parse_mode=HTML отклоняет «голые» <, >, & — ответ модели вроде
    «ВЧД < 20» или «a&b» раньше валил весь sendMessage 400-й ошибкой, и фидбек
    терялся МОЛЧА. Экранируем только подставляемый контент; наши собственные
    теги-шаблоны (<b>, <i>) добавляются ВОКРУГ уже экранированного текста и
    остаются рабочей разметкой. quote=False — кавычки в тексте не трогаем.
    """
    return html.escape(str(text if text is not None else ""), quote=False)


def send_telegram(text: str, disable_notification: bool = False,
                  reply_markup: dict | None = None) -> int | None:
    """Отправить сообщение пользователю через Telegram Bot API.

    Возвращает message_id отправленного сообщения (truthy при успехе) или
    None при ошибке. message_id нужен, чтобы привязать ответ-reply ученика к
    КОНКРЕТНОМУ вопросу (см. reply-привязку в handle_message).

    Если HTML-разметка не распарсилась (битая сущность в неэкранированном
    фрагменте), сообщение НЕ теряется: повторяем отправку как обычный текст
    без parse_mode. Доставка важнее форматирования — это страховка от
    «бот молчит» на любом не пойманном экранированием месте."""
    if not TELEGRAM_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN не задан, пропускаю отправку")
        return None
    user = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    user,
        "text":       text,
        "parse_mode": "HTML",
        "disable_notification": disable_notification,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        r = requests.post(url, json=payload, timeout=10)
        data = r.json()
        if data.get("ok"):
            return data.get("result", {}).get("message_id")
        log.error("telegram send failed: %s", r.text)
        # Падение HTML-парсера (400) → пересылаем без разметки, чтобы не потерять
        # сообщение. Тегов-шаблонов немного, но в plain-тексте они станут видимы
        # буквально — это приемлемая деградация против немой потери фидбека.
        if payload.get("parse_mode"):
            payload.pop("parse_mode", None)
            r2 = requests.post(url, json=payload, timeout=10)
            d2 = r2.json()
            if d2.get("ok"):
                log.warning("telegram: пересылка без parse_mode удалась")
                return d2.get("result", {}).get("message_id")
            log.error("telegram plain resend failed: %s", r2.text)
        return None
    except Exception:
        log.exception("telegram send error")
        return None


def send_telegram_document(filename: str, content: "str | bytes",
                           caption: str = "",
                           mime: str = "text/markdown") -> bool:
    """Отправить файл (текст или PDF) в Telegram."""
    if not TELEGRAM_TOKEN:
        return False
    user = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    blob = content.encode("utf-8") if isinstance(content, str) else content
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"

    def _post(use_html: bool) -> requests.Response:
        data = {"chat_id": user, "caption": caption[:1024],
                "disable_notification": "true"}
        if use_html:
            data["parse_mode"] = "HTML"
        return requests.post(
            url, data=data,
            files={"document": (filename, blob, mime)}, timeout=60)

    try:
        r = _post(use_html=True)
        if r.json().get("ok", False):
            return True
        # Битая HTML-сущность в caption → шлём документ без parse_mode, чтобы
        # сам файл (конспект/алгоритм) точно дошёл.
        log.error("telegram sendDocument failed: %s", r.text)
        r2 = _post(use_html=False)
        ok = r2.json().get("ok", False)
        if ok:
            log.warning("telegram sendDocument: отправка без parse_mode удалась")
        else:
            log.error("telegram sendDocument plain failed: %s", r2.text)
        return ok
    except Exception:
        log.exception("telegram sendDocument error")
        return False


# ── ремедиальная теория (слабый ответ → конспект файлом) ──────────────────────

THEORY_THRESHOLD = 0.5   # ниже этой доли — присылаем конспект


def _safe_filename(topic: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in topic)
    return (keep.strip()[:60] or "тема")


def send_theory(topic: str, *, question: str = "", answer: str = "") -> bool:
    """Сгенерировать конспект по теме и прислать красивым PDF (откат — .md)."""
    from neurotutor.theory import build_theory
    try:
        md = build_theory(topic, question=question, answer=answer)
    except Exception:
        log.exception("theory build failed for %s", topic)
        return False
    if not md:
        return False

    base = _safe_filename(topic)
    caption = f"📘 Конспект по теме: <b>{esc(topic)}</b>"
    try:
        from neurotutor.render import markdown_to_pdf
        pdf = markdown_to_pdf(md, title=topic)
    except Exception:
        log.exception("pdf render failed")
        pdf = None
    if pdf:
        return send_telegram_document(f"Конспект — {base}.pdf", pdf,
                                      caption=caption, mime="application/pdf")
    # откат: если PDF не собрался — отправим markdown-файл
    return send_telegram_document(f"Конспект — {base}.md", md, caption=caption)


# алгоритмические режимы: при ошибке шлём сам эталонный алгоритм, не конспект
ALGO_KINDS = {"surgical_steps", "emergency", "crisis"}


def send_algorithm(topic: str, *, kind: str = "", question: str = "",
                   answer: str = "") -> bool:
    """Сгенерировать эталонный пошаговый алгоритм и прислать PDF (откат — .md)."""
    from neurotutor.theory import build_algorithm
    try:
        md = build_algorithm(topic, kind=kind, question=question, answer=answer)
    except Exception:
        log.exception("algorithm build failed for %s", topic)
        return False
    if not md:
        return False

    base = _safe_filename(topic)
    caption = f"📐 Эталонный алгоритм: <b>{esc(topic)}</b>"
    try:
        from neurotutor.render import markdown_to_pdf
        pdf = markdown_to_pdf(md, title=topic)
    except Exception:
        log.exception("pdf render failed")
        pdf = None
    if pdf:
        return send_telegram_document(f"Алгоритм — {base}.pdf", pdf,
                                      caption=caption, mime="application/pdf")
    return send_telegram_document(f"Алгоритм — {base}.md", md, caption=caption)


# ── inbox reader (file-based, совместим с telegram-bot-agent skill) ───────────


def read_inbox() -> list[dict]:
    """Вернуть список {text, id} из inbox-файла. После чтения не трогаем файл —
    обработка идёт через offset в long_polling."""
    if not INBOX.exists():
        return []
    try:
        with open(INBOX) as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        log.exception("inbox read error")
        return []


def clear_inbox() -> None:
    """Очистить inbox после обработки."""
    try:
        INBOX.write_text("")
    except Exception:
        log.exception("inbox clear failed")


# ── main loop ─────────────────────────────────────────────────────────────────

def send_morning_reminder() -> None:
    """Утреннее напоминание о due-карточках. Вызывать вручную или по крону."""
    due = due_cards_summary()
    if due:
        greeting = (
            "🌅 Доброе утро! Neurotutor на связи.\n"
            f"{due}\n\n"
            "Напиши «карточки» чтобы начать повторение."
        )
        send_telegram(greeting, disable_notification=True)


# ── ежедневный дрилл (замкнутая петля) ────────────────────────────────────────

BLOOM_RU = {1: "запомнить", 2: "понять", 3: "применить",
            4: "анализ", 5: "оценка", 6: "синтез"}

KIND_LABELS = {
    "recall": "🧠 Припоминание",
    "case": "🏥 Клинический случай",
    "surgical_steps": "🔪 Ход операции",
    "emergency": "🚨 Экстренный алгоритм",
    "crisis": "⚡ Интраоп. кризис",
}


def _format_question(q: dict, queue_total: int | None = None) -> str:
    kind = q.get("kind", "recall")
    label = KIND_LABELS.get(kind, "❓ Вопрос")
    if kind == "recall":
        label += f" · {BLOOM_RU.get(q.get('bloom_level', 1), '')}"
    head = f"<b>{label}</b>"
    # Честный счётчик: сколько вопросов ещё в очереди (раньше всегда было «1/N»
    # — нельзя было отличить вопросы друг от друга).
    if queue_total and queue_total > 1:
        head += f"  <i>(в очереди: {queue_total})</i>"
    if kind == "case":
        foot = ("<i>Можешь запрашивать находки: «оцени по Hunt-Hess», "
                "«результат КТ?», «лаб?». Когда готов поставить диагноз — "
                "напиши <b>заключение</b> (или «заключение: …»).</i>")
    else:
        foot = ("<i>Ответь текстом — я оценю. Если вопросов несколько — "
                "ответь <b>reply</b> на нужный, чтобы оценил именно его.</i>")
    return f"{head}\n\n{esc(q['prompt'])}\n\n{foot}"


def _send_next_question(user_id: str | int) -> bool:
    """Отправить следующий неотвеченный вопрос. True если был что слать."""
    pend = drill.open_questions(str(user_id))
    if not pend:
        return False
    # Reset the case "final answer" flag whenever a fresh question is delivered
    # (cron drill, next-in-batch, …). Otherwise a stale True from an abandoned
    # case would make the first exploratory message on the NEW case get graded
    # as the final answer instead of treated as a workup request.
    state = load_session(user_id)
    if state.case_final:
        state.case_final = False
        save_session(user_id, state)
    mid = send_telegram(_format_question(pend[0], queue_total=len(pend)),
                        reply_markup=_case_markup(pend[0]))
    # Запоминаем, каким сообщением доставлен вопрос — чтобы ответ-reply на него
    # оценивался именно как ответ на ЭТОТ вопрос, а не на самый старый в очереди.
    drill.set_question_message(pend[0]["id"], mid)
    return True


# ── выбор режима тренировки (инлайн-кнопки) ───────────────────────────────────

# порядок и подписи кнопок меню режимов (callback_data = "mode:<kind>")
MODE_MENU = [
    ("🧠 Тест (припоминание)", "recall"),
    ("🏥 Клинический случай", "case"),
    ("🔪 Ход операции", "surgical_steps"),
    ("🚨 Экстренный алгоритм", "emergency"),
    ("⚡ Интраоп. кризис", "crisis"),
]

# что открывает меню режимов
MODE_MENU_TRIGGERS = {"/mode", "/menu", "/start", "режим", "режимы",
                      "меню", "старт", "выбор режима", "тренировка"}

# кнопка «финальный ответ» для кейса (то же, что и слово «заключение»)
CASE_FINAL_BTN = {"inline_keyboard": [[
    {"text": "✅ Заключение (финальный ответ)", "callback_data": "case_final"}]]}


def _case_markup(q: dict) -> dict | None:
    """Кнопку финала вешаем только на кейсы (у них интерактивная фаза)."""
    return CASE_FINAL_BTN if q.get("kind") == "case" else None


# триггер финального ответа в кейсе: «заключение», «заключение: …», «мой диагноз».
# Узко — только то, что мы сами афишируем в подсказке/на кнопке. Раньше сюда
# попадали слишком общие «итог»/«мой ответ» и съедали обычный текст разбора
# («итог обследования…» трактовался как финальный ответ).
FINAL_RE = re.compile(
    r"^\s*(?:это\s+|вот\s+)?(?:мо[йёе]\s+)?"
    r"(?:заключение|заключаю|мой\s+диагноз|ставлю\s+диагноз|"
    r"(?:финальн\w*|заключительн\w*|итоговый)\s+ответ)\b[\s:.\-—]*(.*)$",
    re.IGNORECASE | re.DOTALL)

# выход из режима финала обратно в интерактивный разбор кейса
CANCEL_RE = re.compile(
    r"^\s*(?:отмена|назад|стоп|погоди|подожди|продолж\w*)\b",
    re.IGNORECASE)

# явный запрос конспекта: «конспект по аневризмам», «сделай теорию о шунтах»
CONSPECT_RE = re.compile(
    r"^\s*(?:сделай|собери|дай|нужен|нужна|хочу|пришли|скинь|можешь(?:\s+\w+)?)?\s*"
    r"(?:конспект|теори[яю])\b\s*(?:по|на|о[бо]?|про)?\s*(.*)$",
    re.IGNORECASE)

# грубый инференс домена по ключевым словам — чтобы изученный конспектом топик
# попал в нужный срез карты компетенций (расширяет граф за пределы seed).
# Коды строго из db.store.DOMAINS (14 разделов A–M + approaches): ретированные
# pathology/clinical больше не существуют — топик с таким кодом не зарегистрировался
# бы (get_or_create_concept вернул бы None). Порядок: от частного к общему,
# первое совпадение выигрывает; дефолт — anatomy (базовые нейронауки, §A).
_DOMAIN_HINTS = [
    ("vascular", ("аневризм", "авм ", "артериовеноз", "мальформац", "инсульт",
                  "ишеми", "кровоизлия", "сосуд", "каротид", "стеноз сонн",
                  "окклюз", "эмбол", "каверном", "сак", "субарахн")),
    ("hydrocephalus", ("гидроцефал", "ликвор", "вентрикул", "шунт", "etv",
                       "третья вентрикулост")),
    ("spine", ("позвоночник", "спинальн", "спинного", "миелопат", "грыж диск",
               "межпозвон", "спондил", "люмбаль", "цервикальн", "дискэктом")),
    ("trauma", ("травма", "чмт", "гематом", "ушиб мозг", "перелом", "контузи",
                "субдураль", "эпидураль")),
    ("oncology", ("опухол", "глиом", "глиобластом", "менингиом", "аденом",
                  "невринома", "шваннома", "гипофиз", "who", "карцином",
                  "метастаз", "астроцитом", "эпендимом", "основани черепа")),
    ("functional", ("эпилепси", "паркинсон", "dbs", "глубок стимул", "тригеминал",
                    "невралги", "спастичн", "болев синдром", "функциональн")),
    ("pediatric", ("детск", "педиатр", "врожд", "spina bifida", "дизрафи",
                   "краниосиностоз")),
    ("peripheral_nerve", ("периферическ нерв", "карпальн", "туннельн синдром",
                          "сплетени", "плексус", "седалищн", "локтев нерв")),
    ("infection", ("инфекц", "абсцесс", "менингит", "эмпиема", "остеомиелит",
                   "энцефалит", "воспалит")),
    ("neurocritical", ("реанимац", "интенсивн терап", "мониторинг", "внутричерепн",
                       "вчд", "анестези", "наркоз", "седаци", "кризис")),
    ("radiology", ("кт", "мрт", "ангио", "снимок", "визуализац", "dwi",
                   "flair", "перфузи", "трактограф", "нейровизуал")),
    ("approaches", ("доступ", "краниотом", "резекц", "клипир", "операц",
                    "эндоскоп", "трепанац", "оперативн техник")),
    ("anatomy", ("анатом", "цистерн", "тракт", "ядро", "извилин", "артери",
                 "вена", "нерв", "сплетени", "физиолог", "патофизиолог")),
]


def _infer_domain(topic: str) -> str:
    t = topic.lower()
    for domain, words in _DOMAIN_HINTS:
        if any(w in t for w in words):
            return domain
    return "anatomy"


def send_mode_menu() -> None:
    """Прислать инлайн-кнопки выбора режима тренировки."""
    keyboard = {"inline_keyboard": [
        [{"text": label, "callback_data": f"mode:{kind}"}]
        for label, kind in MODE_MENU
    ]}
    send_telegram(
        "🎛 <b>Выбери режим тренировки</b>\n"
        "<i>Тест и кейс при ошибке → конспект из учебников; "
        "алгоритмы → эталонный пошаговый алгоритм.</i>",
        disable_notification=True, reply_markup=keyboard)


def answer_callback_query(callback_id: str) -> None:
    """Снять «часики» с нажатой инлайн-кнопки."""
    if not TELEGRAM_TOKEN or not callback_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_id}, timeout=10)
    except Exception:
        log.exception("answerCallbackQuery failed")


def handle_callback(data: str, user_id: str | int) -> None:
    """Обработать нажатие инлайн-кнопки (выбор режима / финал кейса)."""
    # «✅ Заключение» — перевести открытый кейс в режим финального ответа.
    if data == "case_final":
        if not drill.has_open(str(user_id)):
            send_telegram("Сейчас нет открытого кейса.")
            return
        state = load_session(user_id)
        state.case_final = True
        save_session(user_id, state)
        send_telegram("✍️ Пиши заключение: дифдиагноз, план обследования, тактика.")
        return

    if not data.startswith("mode:"):
        return
    kind = data.split(":", 1)[1]
    label = KIND_LABELS.get(kind, kind)
    # новый вопрос → сбрасываем флаг финала прошлого кейса
    state = load_session(user_id)
    state.case_final = False
    save_session(user_id, state)
    send_telegram(f"🎯 Режим: <b>{label}</b>. Готовлю вопрос…",
                  disable_notification=True)
    q = drill.generate_one(str(user_id), kind)
    if not q:
        send_telegram("Не удалось собрать вопрос этого типа — попробуй другой "
                      "режим (/mode).")
        return
    queue_total = len(drill.open_questions(str(user_id)))
    mid = send_telegram(_format_question(q, queue_total=queue_total),
                        reply_markup=_case_markup(q))
    drill.set_question_message(q["id"], mid)


def send_drill(n: int = 3) -> None:
    """Сгенерировать набор дня (припоминание + сценарии) и отправить первый."""
    uid = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    if not uid:
        log.error("TELEGRAM_USER_ID не задан — некому слать дрилл")
        return
    created = drill.generate_drill(str(uid), n_recall=n)
    if created:
        kinds = ", ".join(sorted({KIND_LABELS.get(c["kind"], c["kind"])
                                  for c in created}))
        send_telegram(
            f"🧠 <b>Разбор дня</b>: {len(created)} задани(й).\n<i>{kinds}</i>\n"
            "Отвечай по одному 👇", disable_notification=True)
    _send_next_question(uid)


def send_reminder() -> None:
    """Вечернее напоминание о неотвеченных вопросах дня (для крона).

    Молчит, если открытых вопросов нет (никакого пустого спама). Сначала
    отрабатывает TTL — про уже истёкшие вопросы не напоминаем.
    """
    uid = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    if not uid:
        log.error("TELEGRAM_USER_ID не задан — некому слать напоминание")
        return
    drill.expire_stale(str(uid))
    pend = drill.open_questions(str(uid))
    if not pend:
        log.info("reminder: открытых вопросов нет, молчу")
        return
    send_telegram(
        f"⏰ Висит без ответа: <b>{len(pend)}</b> вопрос(а) дня. "
        "Неотвеченные сгорают через 48 ч и заменяются новыми.\n"
        "<i>Можно ответить сейчас, «пропусти» — следующий, "
        "«сбрось вопросы» — очистить очередь.</i>",
        disable_notification=True)
    _send_next_question(uid)


def send_regrade() -> None:
    """Переоценить ответы, отложенные из-за перегрузки MiniMax (529), и прислать
    результаты. Бэкстоп замкнутой петли: оценка, упавшая на перегрузе, не
    теряется — только откладывается до восстановления сервиса.

    Молчит, если отложенных ответов нет.
    """
    uid = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    if not uid:
        return
    pend = drill.pending_regrade(str(uid))
    if not pend:
        log.info("regrade: отложенных ответов нет, молчу")
        return
    log.info("regrade: переоцениваю %d отложенн(ых) ответ(а)", len(pend))
    try:
        results = drill.regrade_saved(str(uid))
    except Exception:
        log.exception("regrade pass failed")
        return
    # Доставляем фидбек без авто-выдачи следующего вопроса (чтобы пакет не сыпал
    # вопросами); один следующий открытый вопрос отправим в конце.
    for fb in results:
        _deliver_feedback(str(uid), fb, advance=False)
    if results:
        _send_next_question(uid)


def send_digest(days: int = 14) -> None:
    """Собрать PubMed-дайджест и отправить в Telegram. Для крона/ручного запуска."""
    from neurotutor.digest import build_digest
    try:
        msg = build_digest(days=days)
    except Exception:
        log.exception("digest build failed")
        return
    if msg:
        send_telegram(msg, disable_notification=True)
    else:
        log.info("digest empty — nothing to send")


def send_report() -> None:
    """Отправить ретроспективную карту компетенций. Для крона/ручного запуска."""
    from neurotutor.report import format_report_telegram
    try:
        msg = format_report_telegram()
    except Exception:
        log.exception("report build failed")
        return
    if msg:
        send_telegram(msg, disable_notification=True)


def _handle_evening_report(text: str) -> None:
    """Парсит свободный отчёт за день и пишет в daily_logs.

    Голосовое о дне: «сделал X, застрял на Y, завтра Z, энергия 7/10».
    Триггеры обрабатываются в handle_message.
    """
    import subprocess
    handler = "/root/.hermes/scripts/evening_report_handler.py"
    try:
        result = subprocess.run(
            ["python3", handler, "--text", text, "--quiet"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            send_telegram(
                "✅ <b>Отчёт записан в daily_logs.</b>\n\n"
                "📌 Что сделал — сохранил\n"
                "🚧 Застрял — записал\n"
                "📋 Завтра — в плане\n\n"
                "Хочешь посмотреть весь лог за неделю — скажи «покажи неделю»."
            )
        else:
            log.error(f"evening_report failed: {result.stderr}")
            send_telegram("⚠️ Не смог распарсить отчёт. Скажи ещё раз чуть короче.")
    except Exception:
        log.exception("evening_report crash")
        send_telegram("⚠️ Ошибка парсера. Запишу вручную.")


def _question_quote(fb: dict) -> str:
    """Короткая однострочная цитата оцениваемого вопроса для фидбека.

    Без неё ученик не видел, на какой вопрос пришла оценка, и она казалась
    ответом «не на то сообщение». С цитатой связка «вопрос ↔ оценка» очевидна.
    """
    q_txt = (fb.get("question") or "").strip().replace("\n", " ")
    q_txt = re.sub(r"\s{2,}", " ", q_txt)
    # для кейса выкидываем служебный префикс «Клинический случай.»
    q_txt = re.sub(r"^Клинический случай\.?\s*", "", q_txt)
    if not q_txt:
        return ""
    if len(q_txt) > 90:
        q_txt = q_txt[:90].rstrip() + "…"
    return f"📝 <i>Оценка по вопросу:</i> «{esc(q_txt)}»\n\n"


def _deliver_feedback(user_id: str, fb: dict, *, advance: bool = True) -> None:
    """Доставить результат оценки одного ответа: текст оценки, раскрытие
    диагноза для кейса, ремедиацию при слабом ответе, следующий вопрос.

    advance=False — только фидбек, без выдачи следующего вопроса (используется
    при пакетной переоценке отложенных ответов, чтобы не сыпать вопросами).
    """
    quote = _question_quote(fb)

    # Сервис оценки был перегружен (529): ответ сохранён, вопрос остался за
    # учеником, переоценим автоматически. Никакой немой потери.
    if fb.get("degraded"):
        send_telegram(
            quote + "⏳ Сервис оценки сейчас перегружен. Твой ответ "
            "<b>сохранён</b> — оценю автоматически, как только отпустит. "
            "Вопрос пока остаётся открытым.")
        return

    # Грейдер не смог разобрать ответ — вопрос остался открытым, просим повторить.
    if fb.get("ungraded"):
        send_telegram(quote + "⚠️ Не смог корректно оценить ответ (сбой "
                      "парсинга). Вопрос остался открытым — попробуй "
                      "переформулировать.")
        if advance:
            _send_next_question(user_id)
        return

    answer = fb.get("answer", "")
    score = fb.get("score")
    pct = f"{round((score or 0) * 100)}%"
    mark = "✅" if (score or 0) >= 0.7 else ("🟡" if (score or 0) >= 0.4 else "❌")
    tail = ("\n\n<i>Следующее повторение запланировано.</i>"
            if fb.get("next_review") else "")
    msg = (f"{quote}{mark} <b>Оценка: {pct}</b>\n\n"
           f"{esc(fb.get('feedback', '').strip())}{tail}")
    send_telegram(msg)

    # Кейс: диагноз скрывался в вопросе — теперь, после ответа, раскрываем его.
    if fb.get("kind") == "case" and fb.get("topic"):
        send_telegram(f"🩺 Правильный диагноз: <b>{esc(fb['topic'])}</b>",
                      disable_notification=True)

    # Слабо справился → ремедиация зависит от режима:
    #   алгоритм (ход операции/экстренный/кризис) → эталонный алгоритм;
    #   тест/кейс → структурированный конспект из RAG.
    if (score or 0) < THEORY_THRESHOLD and fb.get("topic"):
        if fb.get("kind") in ALGO_KINDS:
            send_telegram("📐 Покажу эталонный алгоритм по этой теме…",
                          disable_notification=True)
            send_algorithm(fb["topic"], kind=fb["kind"],
                           question=fb.get("question", ""), answer=answer)
        else:
            send_telegram("📘 Подтяну теорию по этой теме — собираю конспект…",
                          disable_notification=True)
            send_theory(fb["topic"], question=fb.get("question", ""), answer=answer)

    if not advance:
        return
    if fb.get("remaining"):
        _send_next_question(user_id)
    else:
        send_telegram("🎉 Все вопросы дня закрыты. Отличная работа!")


def _handle_drill_answer(user_id: str, text: str, qid: int | None = None) -> None:
    """Оценить ответ на висящий вопрос, прислать фидбек и следующий вопрос.

    qid задаёт КОНКРЕТНЫЙ вопрос (reply-привязка); без него оценивается самый
    старый открытый. Если по reply вопрос уже закрыт — мягкий откат к старому.
    """
    fb = (drill.answer_specific(user_id, qid, text) if qid is not None
          else None)
    if fb is None:
        fb = drill.answer_pending(user_id, text)
    if fb is None:
        return
    _deliver_feedback(user_id, fb)


def handle_message(raw: dict, user_id: str | int) -> None:
    """Обработать одно входящее сообщение."""
    text = raw.get("text", "").strip()
    if not text:
        return

    # Открыть меню выбора режима (инлайн-кнопки) — раньше всякой маршрутизации.
    if text.lower().strip() in MODE_MENU_TRIGGERS:
        send_mode_menu()
        return

    state = load_session(user_id)
    routing, persona_hint = route(text)

    # Reply-привязка: если ученик ответил Telegram-reply'ем на конкретный вопрос,
    # узнаём его id и тип — дальше оцениваем/ведём ИМЕННО его, а не самый старый
    # в очереди. Это и есть лечение «отвечает не на то сообщение».
    reply_qid = reply_kind = None
    reply_mid = raw.get("reply_to_message_id")
    if reply_mid:
        rq = drill.open_question_by_message(str(user_id), reply_mid)
        if rq:
            reply_qid, reply_kind = rq["id"], rq["kind"]

    # Пока открыт клинический кейс, сообщение принадлежит кейсу: запрос находки
    # («результат МРТ?») или финальный ответ («…МРТ — измерение желудочков») НЕ
    # должны перекидывать сессию в режим imaging/new по случайному ключевому слову
    # — иначе сообщение минует case_followup / оценку. Сброс и смену персонажа
    # оставляем рабочими.
    _open = drill.open_questions(str(user_id))
    _active_is_case = (reply_kind == "case") if reply_qid is not None else (
        bool(_open) and _open[0].get("kind") == "case")
    if _active_is_case and routing in (
            "diagnostic", "review", "case", "osce", "imaging", "new"):
        routing = ""

    state = apply_route(state, routing, persona_hint)

    if routing == "reset":
        send_telegram("🔄 Сессия сброшена. Напиши, что хочешь учить.")
        clear_session(user_id)
        return

    # Запрос ретроспективной карты компетенций (приоритетнее ответа на вопрос).
    low = text.lower()
    if any(kw in low for kw in ("отчёт", "отчет", "прогресс", "компетенц",
                                 "карта знаний", "мои пробелы", "/report")):
        send_report()
        return

    # Свободный отчёт за день → парсер в daily_logs.
    # Триггеры: "сделал сегодня", "дневной отчёт", "отчитаться", "за день",
    # "/day". Это приоритетнее RAG/кейсов — голосовое о дне не должно
    # ломаться на «сделал» / «выучил» / «прочитал» внутри кейса.
    if any(kw in low for kw in ("сделал сегодня", "дневной отчёт", "дневной отчет",
                                 "отчитаться", "за день", "/day",
                                 "отчет за день", "отчёт за день")):
        _handle_evening_report(text)
        return

    # Калибровка: по одному базовому вопросу на каждый домен программы —
    # засеивает карту компетенций реальным уровнем (холодный старт).
    if re.match(r"^\s*(/calibrate|калибровк\w*)\b", low):
        send_telegram("🧭 Готовлю калибровку: по одному вопросу на каждый из "
                      "14 разделов программы (~2 мин). Текущая очередь "
                      "вопросов сброшена.", disable_notification=True)
        created = drill.generate_calibration(str(user_id))
        if not created:
            send_telegram("Не удалось собрать калибровку — попробуй позже.")
            return
        send_telegram(f"🧭 <b>Калибровка</b>: {len(created)} вопрос(ов), по "
                      "одному на раздел. Отвечай как можешь — «не знаю» тоже "
                      "ответ, это разметка карты, а не экзамен. Можно "
                      "«пропусти». Поехали 👇", disable_notification=True)
        _send_next_question(user_id)
        return

    # Управление очередью вопросов: «пропусти» — сжечь текущий и дать
    # следующий; «сбрось вопросы» — очистить очередь целиком. Без этого
    # неудобный вопрос блокировал всю петлю до TTL.
    if re.match(r"^\s*(пропусти(ть)?|скип|skip)\b", low):
        pend = drill.open_questions(str(user_id))
        if not pend:
            send_telegram("Открытых вопросов нет. Открыть меню — «режим».")
            return
        drill.skip_first(str(user_id))
        send_telegram("⏭ Пропустил (без оценки).", disable_notification=True)
        if not _send_next_question(user_id):
            send_telegram("Очередь пуста. Новый вопрос — через «режим» "
                          "или завтра в утреннем разборе.")
        return
    if re.match(r"^\s*(сбрось|сбросить|очисти(ть)?)\s+(вопрос|очеред)", low):
        n = drill.expire_all_open(str(user_id))
        send_telegram(f"🧹 Очередь очищена ({n} вопрос(а) снято, без оценки). "
                      "Новый вопрос — «режим», или жди утренний разбор.")
        return

    # Явный запрос конспекта по теме → собрать и прислать PDF из RAG.
    m = CONSPECT_RE.match(text)
    if m:
        topic = m.group(1).strip(" .?!:;—-").strip()
        if not topic:
            send_telegram("📘 По какой теме собрать конспект? Напиши, например: "
                          "«конспект по аневризмам ПСА».")
            return
        send_telegram(f"📘 Собираю конспект по теме «<b>{esc(topic)}</b>» из "
                      "учебников… (полминуты)", disable_notification=True)
        if not send_theory(topic):
            send_telegram("Не получилось собрать конспект — уточни тему?")
            return
        # Регистрируем изученный топик как концепт: он войдёт в карту
        # компетенций как «изучено, но не проверено» (пробел до проверки).
        try:
            from neurotutor.db.store import get_or_create_concept
            get_or_create_concept(topic, _infer_domain(topic),
                                  summary="Изучено по конспекту (ещё не проверено)")
        except Exception:
            log.exception("concept registration failed for %s", topic)
        return

    # Если у пользователя висит вопрос дня и это не явная смена режима —
    # трактуем сообщение как ответ на него (замкнутая петля).
    open_qs = drill.open_questions(str(user_id))
    if not routing and open_qs:
        # На какой вопрос отвечаем: reply → конкретный; иначе самый старый.
        # active_qid=None означает «самый старый» — точное прежнее поведение.
        if reply_qid is not None:
            active_kind, active_qid = reply_kind, reply_qid
        else:
            active_kind, active_qid = open_qs[0].get("kind"), None
        # Кейс ведётся интерактивно: запросы находок/шкал — пока не сказано
        # «заключение». Так ординатор сам ставит диагноз, а не получает подсказку.
        if active_kind == "case":
            # Передумал ставить заключение → назад в интерактивный разбор.
            if state.case_final and CANCEL_RE.match(text):
                state.case_final = False
                save_session(user_id, state)
                send_telegram("↩️ Ок, продолжаем разбор. Запрашивай находки "
                              "или напиши «заключение», когда будешь готов.")
                return
            m_fin = FINAL_RE.match(text)
            if m_fin:
                rest = m_fin.group(1).strip()
                if rest:                       # «заключение: <разбор>» → сразу оценка
                    state.case_final = False
                    save_session(user_id, state)
                    _handle_drill_answer(str(user_id), rest, qid=active_qid)
                    return
                state.case_final = True        # «заключение» отдельно → ждём разбор
                save_session(user_id, state)
                send_telegram("✍️ Пиши заключение: дифдиагноз, план "
                              "обследования, тактика.")
                return
            if not state.case_final:           # исследовательский запрос по кейсу
                reply = drill.case_followup(str(user_id), text, qid=active_qid)
                body = reply or ("Не понял запрос — уточни (шкала? "
                                 "обследование?).")
                # Кнопка финала висит на КАЖДОМ шаге разбора: оценку можно
                # запустить в один тап, не угадывая слово-триггер.
                send_telegram(
                    esc(body) + "\n\n<i>Готов? Жми «✅ Заключение» или напиши "
                    "«заключение: …».</i>",
                    reply_markup=CASE_FINAL_BTN)
                return
            state.case_final = False           # был режим финала → это и есть ответ
            save_session(user_id, state)
        _handle_drill_answer(str(user_id), text, qid=active_qid)
        return

    # строим history для CLI
    # предыдущие assistant-реплики сохраняем (user уже в текущем сообщении,
    # state.history содержит всё кроме текущего)
    history_for_cli = list(state.history)

    # обновляем step
    state.step += 1

    # отправляем в Neurotutor
    reply_text = neurotutor_reply(
        mode=state.mode,
        user_message=text,
        persona=state.persona,
        history=history_for_cli,
    )

    # дописываем в историю
    state.history.append({"role": "user",      "content": text})
    state.history.append({"role": "assistant",  "content": reply_text})
    # храним только последние MAX_HISTORY*2 (туда-обратно)
    state.history = state.history[-(MAX_HISTORY * 2):]

    save_session(user_id, state)

    send_telegram(esc(reply_text))


def main() -> None:
    log.info("Telegram-интеграция Neurotutor запущена")

    # проверка конфига
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN не задан в .env")
        sys.exit(1)
    if not TELEGRAM_USER:
        log.warning("TELEGRAM_USER_ID не задан — отправка работать не будет")

    last_offset = 0

    while True:
        try:
            # long-polling через getUpdates
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": last_offset + 1, "timeout": 25},
                timeout=30,
            )
            data = r.json()
            if not data.get("ok"):
                log.error("getUpdates error: %s", data)
                time.sleep(5)
                continue

            updates = data.get("result", [])
            for upd in updates:
                last_offset = upd["update_id"]

                # нажатие инлайн-кнопки (выбор режима)
                if "callback_query" in upd:
                    cq = upd["callback_query"]
                    cq_from = str(cq.get("from", {}).get("id", ""))
                    if TELEGRAM_USER and cq_from != str(TELEGRAM_USER):
                        continue
                    answer_callback_query(cq.get("id", ""))
                    cdata = cq.get("data", "")
                    log.info("⮞ callback [%s] %s", cq_from or "?", cdata)
                    handle_callback(cdata, user_id=cq_from or 0)
                    continue

                msg = upd.get("message", {})
                # проверяем user_id
                msg_from = str(msg.get("from", {}).get("id", ""))
                if TELEGRAM_USER and msg_from != str(TELEGRAM_USER):
                    log.debug("ignoring message from %s (not our user)", msg_from)
                    continue

                text = msg.get("text", "").strip()
                if not text:
                    continue

                reply_to = (msg.get("reply_to_message") or {}).get("message_id")
                log.info("→ [%s] %s%s", msg_from or "?", text[:60],
                         f" (reply→{reply_to})" if reply_to else "")
                handle_message({"text": text, "id": msg.get("message_id"),
                                "reply_to_message_id": reply_to},
                               user_id=msg_from or 0)

        except requests.exceptions.Timeout:
            log.debug("long-poll timeout, retrying")
        except Exception:
            log.exception("polling error")
            time.sleep(5)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--morning":
        send_morning_reminder()
    elif arg == "--drill":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        send_drill(n)
    elif arg == "--digest":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        send_digest(days)
    elif arg == "--report":
        send_report()
    elif arg == "--remind":
        send_reminder()
    elif arg == "--regrade":
        send_regrade()
    elif arg == "--menu":
        send_mode_menu()
    elif arg == "--theory":
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if topic:
            send_theory(topic)
        else:
            print("usage: tg_handler.py --theory <тема>")
    else:
        main()