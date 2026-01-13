"""SQLite layer: builds the database, parses sql/queries.sql and runs queries.

User-supplied SQL runs through `run_readonly`, which enforces read-only access
at the SQLite level (an authorizer callback plus `PRAGMA query_only`), not only
by checking the query text.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_loader import DATA_DIR, IBM_COLUMNS, PROJECT_ROOT

QUERIES_PATH = PROJECT_ROOT / "sql" / "queries.sql"
DB_PATH = DATA_DIR / "hr_attrition.db"
TABLE_NAME = "employees"

# Limits for user queries.
MAX_ROWS = 1_000
TIMEOUT_SECONDS = 3.0

# Authorizer action codes that only read data. Everything else is denied.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # recursive CTEs
}


class UnsafeQueryError(ValueError):
    """Raised when a user query is not a single read-only SELECT."""


@dataclass(frozen=True)
class SqlQuery:
    """One named query from sql/queries.sql."""

    name: str
    title: str
    description: str
    sql: str


# --------------------------------------------------------------------------- #
# Query file parsing
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)


def parse_queries(path: Path = QUERIES_PATH) -> dict[str, SqlQuery]:
    """Parse the `-- name:` blocks in a .sql file into SqlQuery objects."""
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_HEADER_RE.finditer(text))
    queries: dict[str, SqlQuery] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():end]
        title = _meta(block, "title") or m.group(1)
        desc = _meta(block, "desc") or ""
        # Body = everything that is not a metadata comment line.
        body = "\n".join(
            line for line in block.strip().splitlines()
            if not re.match(r"^--\s*(title|desc):", line)
        ).strip()
        queries[m.group(1)] = SqlQuery(m.group(1), title, desc, body)
    return queries


def _meta(block: str, key: str) -> str | None:
    m = re.search(rf"^--\s*{key}:\s*(.+)$", block, re.MULTILINE)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# Database construction
# --------------------------------------------------------------------------- #
def to_sql_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only raw IBM columns so SQL derives its own bands."""
    cols = [c for c in IBM_COLUMNS if c in df.columns]
    return df[cols].copy()


def build_database(df: pd.DataFrame, db_path: Path = DB_PATH, overwrite: bool = False) -> Path:
    """Write the full dataset to an on-disk SQLite file (built on first run)."""
    db_path = Path(db_path)
    if db_path.exists() and not overwrite:
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        to_sql_frame(df).to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dept ON {TABLE_NAME}(Department)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_role ON {TABLE_NAME}(JobRole)")
    return db_path


def memory_connection(df: pd.DataFrame) -> sqlite3.Connection:
    """In-memory database holding `df` as the employees table.

    The app uses this for filtered data so every query respects the sidebar.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    to_sql_frame(df).to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    return conn


def run_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a trusted query (from queries.sql) and return a DataFrame."""
    return pd.read_sql_query(sql, conn, params=params)


def run_named(conn: sqlite3.Connection, name: str, queries: dict[str, SqlQuery] | None = None) -> pd.DataFrame:
    """Run a query from queries.sql by name."""
    queries = queries or parse_queries()
    if name not in queries:
        raise KeyError(f"No query named '{name}' in {QUERIES_PATH.name}")
    return run_query(conn, queries[name].sql)


# --------------------------------------------------------------------------- #
# Read-only user queries
# --------------------------------------------------------------------------- #
def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_select(sql: str) -> str:
    """Check a user query is one SELECT/WITH statement. Returns the cleaned SQL.

    This is a fast, friendly first check. The real guarantee is the SQLite
    authorizer in `run_readonly`.
    """
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Query is empty.")
    if ";" in cleaned:
        raise UnsafeQueryError("Only one statement is allowed.")
    first_word = cleaned.split(None, 1)[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise UnsafeQueryError("Only SELECT queries (optionally starting with WITH) are allowed.")
    return cleaned


def _readonly_authorizer(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_readonly(
    conn: sqlite3.Connection,
    sql: str,
    max_rows: int = MAX_ROWS,
    timeout: float = TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, bool]:
    """Run an untrusted query with read-only, row and time limits.

    Returns (result, truncated) where truncated is True if more than
    `max_rows` rows were produced.
    """
    cleaned = validate_select(sql)
    deadline = time.monotonic() + timeout

    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_readonly_authorizer)
    # A non-zero return aborts the running statement.
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(cleaned)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        if "interrupted" in msg:
            msg = f"Query exceeded the {timeout:.0f}s time limit."
        elif "not authorized" in msg:
            msg = "That operation is not allowed. This box is read-only."
        raise UnsafeQueryError(msg) from exc
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)
    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=cols), truncated
