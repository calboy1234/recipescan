"""
init_db.py — RecipeScan database initialisation

Runs automatically on container start. Safe to re-run on an existing
database — uses CREATE TABLE IF NOT EXISTS and ALTER TABLE migrations.
"""

import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "/data/database/recipescan.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path    TEXT UNIQUE NOT NULL,
    file_hash    TEXT UNIQUE NOT NULL,
    is_reviewed  INTEGER DEFAULT 0,
    added_at     TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS ocr_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id           INTEGER NOT NULL,
    engine             TEXT NOT NULL,
    text               TEXT,
    confidence         TEXT,
    recipe_score       REAL DEFAULT 0.0,
    signals            TEXT,
    rotation_corrected INTEGER DEFAULT 0,
    created_at         TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS ocr_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    processed   INTEGER DEFAULT 0,
    skipped     INTEGER DEFAULT 0,
    errors      INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'running'
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS ocr_log_lines (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    line   TEXT,
    ts     TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES ocr_runs(id)
)
""")

# ── Migrations for existing databases ─────────────────────────────────────────
_img_cols = {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}
if "is_reviewed" not in _img_cols:
    cur.execute("ALTER TABLE images ADD COLUMN is_reviewed INTEGER DEFAULT 0")
    print("Migration: added 'is_reviewed' column to images.")

# ── Indexes ───────────────────────────────────────────────────────────────────
cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_image_id  ON ocr_results(image_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_score     ON ocr_results(recipe_score)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_img_reviewed  ON images(is_reviewed)")

# ── Default settings ──────────────────────────────────────────────────────────
cur.executemany(
    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
    [
        ("recipe_threshold", "0.75"),
    ]
)

conn.commit()
conn.close()

print("RecipeScan database initialised successfully.")
