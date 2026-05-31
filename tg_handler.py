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

import json
import logging
import os
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
LIN_KW        = {"персонаж линь", "лин", "май линь", "/lin"}


def route(text: str) -> tuple[str, str]:
    """Вернуть (mode, persona). text в lowercase."""
    t = text.lower().strip()

    # сначала сброс — он важнее всего
    if any(p in t for p in RESET_PHRASES):
        return "reset", ""

    # персонажи (не меняют mode)
    if any(p in t for p in CORVIN_KW):
        return "persona:corvin", "corvin"
    if any(p in t for p in LIN_KW):
        return "persona:lin", "lin"

    # режимы
    if any(p in t for p in DIAGNOSTIC_KW):
        return "diagnostic", ""
    if any(p in t for p in REVIEW_KW):
        return "review", ""
    if any(p in t for p in CASE_KW):
        return "case", ""
    if any(p in t for p in OSCE_KW):
        return "osce", ""
    if any(p in t for p in IMAGING_KW):
        return "imaging", ""
    if any(p in t for p in NEW_KW):
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
    """
    Вызывает Neurotutor CLI с историей.

    history — список dict с ключами role/content (.messages[-MAX_HISTORY:]).
    Преобразуем в текстовую историю для CLI-команды (history пока не
    поддерживается через аргументы, поэтому склеиваем в строку).
    """
    # собираем history как текст
    hist_block = ""
    if history:
        lines = []
        for h in history[-MAX_HISTORY:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if content:
                lines.append(f"{role.title()}: {content}")
        if lines:
            hist_block = "\n".join(lines) + "\n\n"

    full_prompt = f"{hist_block}Пользователь: {user_message}"

    # работаем в директории проекта
    cmd = [
        sys.executable, "-m", "neurotutor.cli", "ask",
        mode, full_prompt,
        "--persona", persona,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BASE_DIR),
        )
        if result.returncode != 0:
            log.error("CLI error: %s", result.stderr)
            return f"[ошибка CLI: {result.stderr.strip()}]"
        # output может содержать rich-разметку — убираем ttags
        out = result.stdout.strip()
        # убираем строки-артефакты от rich
        lines = [l for l in out.splitlines()
                 if not l.startswith("\x1b[") and "tool " not in l.lower()]
        return "\n".join(lines).strip()
    except subprocess.TimeoutExpired:
        return "[тайм-аут 120 сек — попробуй ещё раз]"
    except Exception:
        log.exception("neurotutor reply failed")
        return "[внутренняя ошибка, попробуй позже]"


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


def send_telegram(text: str, disable_notification: bool = False) -> bool:
    """Отправить сообщение пользователю через Telegram Bot API."""
    if not TELEGRAM_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN не задан, пропускаю отправку")
        return False
    user = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id":    user,
                "text":       text,
                "parse_mode": "HTML",
                "disable_notification": disable_notification,
            },
            timeout=10,
        )
        ok = r.json().get("ok", False)
        if not ok:
            log.error("telegram send failed: %s", r.text)
        return ok
    except Exception:
        log.exception("telegram send error")
        return False


def send_telegram_document(filename: str, content: str,
                           caption: str = "") -> bool:
    """Отправить текстовый файл (конспект теории) в Telegram."""
    if not TELEGRAM_TOKEN:
        return False
    user = TELEGRAM_USER or os.getenv("TELEGRAM_USER_ID", "")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
            data={"chat_id": user, "caption": caption[:1024],
                  "parse_mode": "HTML", "disable_notification": "true"},
            files={"document": (filename, content.encode("utf-8"),
                                "text/markdown")},
            timeout=30,
        )
        ok = r.json().get("ok", False)
        if not ok:
            log.error("telegram sendDocument failed: %s", r.text)
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
    """Сгенерировать конспект по теме и прислать файлом."""
    from neurotutor.theory import build_theory
    try:
        md = build_theory(topic, question=question, answer=answer)
    except Exception:
        log.exception("theory build failed for %s", topic)
        return False
    if not md:
        return False
    fname = f"Теория — {_safe_filename(topic)}.md"
    return send_telegram_document(
        fname, md, caption=f"📘 Конспект по теме: <b>{topic}</b>")


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


def _format_question(q: dict, idx: int | None = None, total: int | None = None) -> str:
    kind = q.get("kind", "recall")
    label = KIND_LABELS.get(kind, "❓ Вопрос")
    if kind == "recall":
        label += f" · {BLOOM_RU.get(q.get('bloom_level', 1), '')}"
    head = f"<b>{label}</b>"
    if idx is not None and total is not None:
        head += f"  <i>({idx}/{total})</i>"
    return f"{head}\n\n{q['prompt']}\n\n<i>Ответь текстом — я оценю.</i>"


def _send_next_question(user_id: str | int) -> bool:
    """Отправить следующий неотвеченный вопрос. True если был что слать."""
    pend = drill.open_questions(str(user_id))
    if not pend:
        return False
    send_telegram(_format_question(pend[0], 1, len(pend)))
    return True


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


def _handle_drill_answer(user_id: str, text: str) -> None:
    """Оценить ответ на висящий вопрос, прислать фидбек и следующий вопрос."""
    fb = drill.answer_pending(user_id, text)
    if fb is None:
        return
    score = fb.get("score")
    pct = f"{round((score or 0) * 100)}%"
    mark = "✅" if (score or 0) >= 0.7 else ("🟡" if (score or 0) >= 0.4 else "❌")
    tail = ("\n\n<i>Следующее повторение запланировано.</i>"
            if fb.get("next_review") else "")
    msg = f"{mark} <b>Оценка: {pct}</b>\n\n{fb.get('feedback','').strip()}{tail}"
    send_telegram(msg)

    # Слабо справился → присылаем структурированный конспект по теме.
    if (score or 0) < THEORY_THRESHOLD and fb.get("topic"):
        send_telegram("📘 Подтяну теорию по этой теме — собираю конспект…",
                      disable_notification=True)
        send_theory(fb["topic"], question=fb.get("question", ""), answer=text)

    if fb.get("remaining"):
        _send_next_question(user_id)
    else:
        send_telegram("🎉 Все вопросы дня закрыты. Отличная работа!")


def handle_message(raw: dict, user_id: str | int) -> None:
    """Обработать одно входящее сообщение."""
    text = raw.get("text", "").strip()
    if not text:
        return

    state = load_session(user_id)
    routing, persona_hint = route(text)
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

    # Если у пользователя висит вопрос дня и это не явная смена режима —
    # трактуем сообщение как ответ на него (замкнутая петля).
    if not routing and drill.has_open(str(user_id)):
        _handle_drill_answer(str(user_id), text)
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

    send_telegram(reply_text)


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
                msg = upd.get("message", {})
                # проверяем user_id
                msg_from = str(msg.get("from", {}).get("id", ""))
                if TELEGRAM_USER and msg_from != str(TELEGRAM_USER):
                    log.debug("ignoring message from %s (not our user)", msg_from)
                    continue

                text = msg.get("text", "").strip()
                if not text:
                    continue

                log.info("→ [%s] %s", msg_from or "?", text[:60])
                handle_message({"text": text, "id": msg.get("message_id")},
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
    elif arg == "--theory":
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if topic:
            send_theory(topic)
        else:
            print("usage: tg_handler.py --theory <тема>")
    else:
        main()