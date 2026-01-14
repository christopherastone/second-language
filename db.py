import sqlite3
from pathlib import Path

from flask import g

from config import get_database_path


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        db_path = get_database_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(exception: Exception | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()
