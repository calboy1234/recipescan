"""
init_db.py — RecipeScan database initialisation

Safe to re-run on an existing database.
Uses CREATE TABLE IF NOT EXISTS + ALTER TABLE migrations so existing data
is never destroyed.

New in v2:
  - failed_images  : tracks every image that errored, enabling targeted retry
  - ocr_runs.total / current_count : enables live progress bars
  - ocr_runs.options : JSON blob of per-run configuration
  - ocr_runs status values: 'running' | 'complete' | 'error' | 'stopped' | 'interrupted'
  - Composite index on ocr_log_lines(run_id, id) for fast SSE streaming
  - Additional default settings: worker_count, batch_size, log_retention_runs, commit_every
  - Stale-run recovery: any row left as 'running' on startup is marked 'interrupted'
  - PRAGMA optimize for long-running deployments
"""

import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "/data/database/recipescan.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys = ON")
cur = conn.cursor()

# ── Core tables ────────────────────────────────────────────────────────────────

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
    osd_confidence     REAL DEFAULT 0.0,
    word_confidence    REAL DEFAULT 0.0,
    psm                INTEGER DEFAULT 3,
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
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at   TEXT,
    processed     INTEGER DEFAULT 0,
    skipped       INTEGER DEFAULT 0,
    errors        INTEGER DEFAULT 0,
    total         INTEGER DEFAULT 0,
    current_count INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'running',
    options       TEXT DEFAULT '{}'
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

# ── New in v2: failed image tracking ──────────────────────────────────────────
#
# Every image that raises an exception during OCR is recorded here.
# The admin UI can trigger a "retry errors" run that re-queues these images,
# bypassing the normal hash-based skip logic so they get a second attempt.

cur.execute("""
CREATE TABLE IF NOT EXISTS failed_images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL,
    file_path  TEXT NOT NULL,
    file_hash  TEXT,
    error_msg  TEXT,
    retried    INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES ocr_runs(id)
)
""")

# ── Migrations: add columns that may be missing in older installs ──────────────

_img_cols = {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}
if "is_reviewed" not in _img_cols:
    cur.execute("ALTER TABLE images ADD COLUMN is_reviewed INTEGER DEFAULT 0")
    print("Migration: added 'is_reviewed' to images.")

if "captured_at" not in _img_cols:
    cur.execute("ALTER TABLE images ADD COLUMN captured_at TEXT")
    cur.execute("UPDATE images SET captured_at = added_at WHERE captured_at IS NULL")
    print("Migration: added 'captured_at' to images and populated from added_at.")

_ocr_cols = {row[1] for row in cur.execute("PRAGMA table_info(ocr_results)").fetchall()}
if "ocr_confidence" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN ocr_confidence REAL DEFAULT 0.0")
    print("Migration: added 'ocr_confidence' to ocr_results.")

# Rename confidence → osd_confidence (requires SQLite 3.25+, shipped with Python 3.6+)
# osd_confidence stores rotation-detection confidence from image_to_osd(), not OCR quality.
if "confidence" in _ocr_cols and "osd_confidence" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results RENAME COLUMN confidence TO osd_confidence")
    print("Migration: renamed 'confidence' to 'osd_confidence' in ocr_results.")

# Rename ocr_confidence → word_confidence for clarity
# word_confidence is the averaged per-word recognition score (0–100) from image_to_data().
if "ocr_confidence" in _ocr_cols and "word_confidence" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results RENAME COLUMN ocr_confidence TO word_confidence")
    print("Migration: renamed 'ocr_confidence' to 'word_confidence' in ocr_results.")
elif "word_confidence" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN word_confidence REAL DEFAULT 0.0")
    print("Migration: added 'word_confidence' to ocr_results.")

# osd_confidence was previously stored as TEXT ("3.45"); migrate to REAL silently —
# SQLite will cast on read so no explicit conversion is needed, but future rows
# are written as REAL directly.
if "osd_confidence" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN osd_confidence REAL DEFAULT 0.0")
    print("Migration: added 'osd_confidence' to ocr_results.")

if "psm" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN psm INTEGER DEFAULT 3")
    print("Migration: added 'psm' to ocr_results (existing rows default to 3).")

if "oem" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN oem INTEGER DEFAULT 3")
    print("Migration: added 'oem' to ocr_results (existing rows default to 3).")

if "run_id" not in _ocr_cols:
    cur.execute("ALTER TABLE ocr_results ADD COLUMN run_id INTEGER REFERENCES ocr_runs(id)")
    print("Migration: added 'run_id' to ocr_results.")

_run_cols = {row[1] for row in cur.execute("PRAGMA table_info(ocr_runs)").fetchall()}
if "total" not in _run_cols:
    cur.execute("ALTER TABLE ocr_runs ADD COLUMN total INTEGER DEFAULT 0")
    print("Migration: added 'total' to ocr_runs.")
if "current_count" not in _run_cols:
    cur.execute("ALTER TABLE ocr_runs ADD COLUMN current_count INTEGER DEFAULT 0")
    print("Migration: added 'current_count' to ocr_runs.")
if "options" not in _run_cols:
    cur.execute("ALTER TABLE ocr_runs ADD COLUMN options TEXT DEFAULT '{}'")
    print("Migration: added 'options' to ocr_runs.")

# ── Indexes ────────────────────────────────────────────────────────────────────

cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_image_id   ON ocr_results(image_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_score      ON ocr_results(recipe_score)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_img_reviewed   ON images(is_reviewed)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_img_hash       ON images(file_hash)")

# Composite index: critical for SSE streaming (WHERE run_id=? AND id>?)
cur.execute("CREATE INDEX IF NOT EXISTS idx_log_run_id     ON ocr_log_lines(run_id, id)")

cur.execute("CREATE INDEX IF NOT EXISTS idx_failed_run     ON failed_images(run_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_failed_retried ON failed_images(retried)")

# ── Default settings ───────────────────────────────────────────────────────────

cur.executemany(
    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
    [
        ("recipe_threshold",    "0.60"),
        # Number of parallel OCR threads.
        ("worker_count",        "2"),
        # Max futures submitted at once. Caps memory usage at scale.
        # 200 futures × ~a few MB each = well under 1 GB for large runs.
        ("batch_size",          "200"),
        # Commit to SQLite every N completions (lower = safer, higher = faster).
        ("commit_every",        "50"),
        # Only keep log lines for the last N completed runs; older ones are pruned.
        ("log_retention_runs",  "10"),
    ]
)

# ── Stale-run recovery ─────────────────────────────────────────────────────────
#
# If the container was killed or crashed mid-run, the ocr_runs row is left
# as 'running'. Detect and mark those as 'interrupted' on every startup so
# the UI never shows a phantom running state and users know to retry.

interrupted = cur.execute(
    "UPDATE ocr_runs "
    "SET status='interrupted', finished_at=CURRENT_TIMESTAMP "
    "WHERE status='running'"
).rowcount
if interrupted:
    print(f"Recovery: marked {interrupted} stale run(s) as 'interrupted'.")

# ── DB maintenance ─────────────────────────────────────────────────────────────

conn.execute("PRAGMA optimize")

conn.commit()
conn.close()

print("RecipeScan database initialised successfully.")
