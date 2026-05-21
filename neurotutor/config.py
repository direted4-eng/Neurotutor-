from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    minimax_api_key: str
    minimax_group_id: str
    minimax_base_url: str
    text_model: str
    vision_model: str
    embed_model: str

    db_path: Path
    rag_dir: Path
    log_level: str

    pubmed_email: str
    pubmed_api_key: str


def load_settings() -> Settings:
    db_path = Path(os.getenv("NEUROTUTOR_DB", "./data/neurotutor.sqlite"))
    rag_dir = Path(os.getenv("NEUROTUTOR_RAG_DIR", "./data/rag"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rag_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
        minimax_group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        minimax_base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        text_model=os.getenv("MINIMAX_TEXT_MODEL", "MiniMax-Text-01"),
        vision_model=os.getenv("MINIMAX_VISION_MODEL", "MiniMax-VL-01"),
        embed_model=os.getenv("MINIMAX_EMBED_MODEL", "embo-01"),
        db_path=db_path,
        rag_dir=rag_dir,
        log_level=os.getenv("NEUROTUTOR_LOG_LEVEL", "INFO"),
        pubmed_email=os.getenv("PUBMED_EMAIL", ""),
        pubmed_api_key=os.getenv("PUBMED_API_KEY", ""),
    )


SETTINGS = load_settings()
