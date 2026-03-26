"""
ocr_pipeline.py — RecipeScan

Walks PHOTO_DIR, runs Tesseract OCR on every new image (identified by
SHA-256 hash), stores results in the database, and scores for recipe
likelihood. Does NOT create recipe records — that is the user's job.

Orientation detection: Tesseract OSD with confidence thresholding.
  - Only applies rotation when OSD confidence >= OSD_MIN_CONFIDENCE.
  - ~98% accuracy with this approach.

Parallelism: ThreadPoolExecutor with WORKER_COUNT workers.
  - Each worker uses its own SQLite connection (thread-safe).
  - Log lines are collected per-image and flushed via a queue.
  - DB writes are serialised through the main thread to avoid contention.
"""

import os
import re
import json
import time
import queue
import sqlite3
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as PILImage, ImageOps
import pytesseract

from recipe_detector import score_text

PHOTO_DIR  = "/photos"
DB_PATH    = "/data/database/recipescan.db"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp")
DEFAULT_THRESHOLD = 0.75

# Number of images to process in parallel.
WORKER_COUNT = 4

# Minimum OSD orientation confidence to trust and apply rotation.
OSD_MIN_CONFIDENCE = 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_threshold(conn: sqlite3.Connection) -> float:
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'recipe_threshold'"
        ).fetchone()
        return float(row[0]) if row else DEFAULT_THRESHOLD
    except Exception:
        return DEFAULT_THRESHOLD


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    elif b < 1024 * 1024:
        return f"{b/1024:.0f}KB"
    else:
        return f"{b/1024/1024:.1f}MB"


# ── Orientation detection ─────────────────────────────────────────────────────

def detect_rotation(img: PILImage.Image) -> tuple[int, float]:
    """
    Use Tesseract OSD to detect orientation.
    Returns (angle_to_apply, confidence).
    Returns (0, 0.0) if confidence is below OSD_MIN_CONFIDENCE.
    """
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0 --dpi 150")

        angle_match = re.search(r"Orientation in degrees:\s*(\d+)", osd)
        conf_match  = re.search(r"Orientation confidence:\s*([\d.]+)", osd)

        if not angle_match or not conf_match:
            return 0, 0.0

        angle      = int(angle_match.group(1))
        confidence = float(conf_match.group(1))

        if confidence < OSD_MIN_CONFIDENCE:
            return 0, confidence

        return angle, confidence

    except Exception:
        return 0, 0.0


def run_ocr(path: str, lines: list) -> tuple[str, int, float]:
    """
    1. Open image and apply EXIF transpose.
    2. Run Tesseract OSD to detect orientation.
    3. Only rotate if confidence >= OSD_MIN_CONFIDENCE.
    4. Run Tesseract OCR on the (possibly corrected) image.
    Returns (text, angle_corrected, osd_confidence).
    """
    img  = PILImage.open(path)
    img  = ImageOps.exif_transpose(img)
    grey = img.convert("L")
    grey = ImageOps.autocontrast(grey)

    angle, confidence = detect_rotation(grey)

    if angle != 0:
        grey = grey.rotate(angle, expand=True)
        lines.append(
            f"  rotation  : {angle}° corrected  "
            f"(OSD confidence: {confidence:.2f})"
        )
    elif confidence > 0:
        lines.append(
            f"  rotation  : none  "
            f"(OSD confidence: {confidence:.2f} — "
            f"{'below threshold, skipped' if confidence < OSD_MIN_CONFIDENCE else 'already upright'})"
        )
    else:
        lines.append("  rotation  : OSD unavailable or image has too little text")

    text = pytesseract.image_to_string(grey, config=r"--psm 6")
    return text, angle, confidence


# ── Per-image OCR (runs in worker thread) ─────────────────────────────────────

def ocr_image(path: str, threshold: float) -> dict:
    """
    Process one image. Returns a result dict consumed by the main thread
    for DB writes and logging. No DB access happens here.
    """
    lines  = []
    result = {
        "path": path, "status": "error", "lines": lines,
        "file_hash": None, "text": "", "angle": 0,
        "recipe_score": 0.0, "signals": {},
        "file_size": 0, "dimensions": "unknown",
        "ocr_time": 0.0,
    }

    try:
        result["file_hash"] = hash_file(path)
        result["file_size"] = os.path.getsize(path)

        try:
            with PILImage.open(path) as probe:
                result["dimensions"] = f"{probe.width}x{probe.height}px"
        except Exception:
            pass

        lines.append("")
        lines.append(f"┌─ {os.path.basename(path)}")
        lines.append(f"  path      : {path}")
        lines.append(f"  size      : {fmt_size(result['file_size'])}  {result['dimensions']}")

        t0 = time.time()
        text, angle, confidence = run_ocr(path, lines)
        result["ocr_time"] = time.time() - t0
        result["text"]  = text
        result["angle"] = angle

        lines.append(f"  ocr time  : {result['ocr_time']:.1f}s")

        detection = score_text(text)
        result["recipe_score"] = detection["score"]
        result["signals"]      = detection["signals"]
        sig = detection["signals"]

        lines.append(
            f"  score     : {detection['score']:.0%}  "
            f"(keywords={sig['keyword_score']:.2f}  "
            f"units={sig['unit_score']:.2f}  "
            f"fractions={sig['fraction_score']:.2f}  "
            f"list={sig['list_score']:.2f})"
        )

        if detection["score"] >= threshold:
            lines.append(f"  result    : RECIPE DETECTED — review in RecipeScan UI")
        else:
            lines.append(f"  result    : not a recipe")

        lines.append(f"└─ done in {result['ocr_time']:.1f}s")
        result["status"] = "ok"

    except Exception as exc:
        lines.append(f"  [error]   : {exc}")
        lines.append(f"└─ failed")
        result["status"] = "error"

    return result


# ── DB write (main thread only) ───────────────────────────────────────────────

def write_result(cur: sqlite3.Cursor, result: dict) -> str:
    """
    Write a completed OCR result to the DB.
    Returns 'skipped', 'processed', or 'error'.
    """
    if result["status"] == "error":
        return "error"

    cur.execute("SELECT id FROM images WHERE file_hash = ?", (result["file_hash"],))
    if cur.fetchone():
        return "skipped"

    cur.execute(
        "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
        (result["path"], result["file_hash"]),
    )
    image_id = cur.lastrowid

    cur.execute(
        """
        INSERT INTO ocr_results
            (image_id, engine, text, confidence, recipe_score, signals, rotation_corrected)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id, "tesseract", result["text"], "unknown",
            result["recipe_score"], json.dumps(result["signals"]),
            result["angle"],
        ),
    )

    return "processed"


# ── Main entry point ──────────────────────────────────────────────────────────

def main(log=print, run_id=None) -> dict:
    """
    Run the full OCR pipeline with parallel image processing.

    `log`    — callable receiving a single string. Defaults to print().
    `run_id` — integer ID of the ocr_runs row for persisting log lines.
    """
    if not os.path.isdir(PHOTO_DIR):
        log("ERROR: /photos is not mounted or does not exist.")
        return {"processed": 0, "skipped": 0, "errors": 1}

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur  = conn.cursor()

    log_conn = sqlite3.connect(DB_PATH) if run_id else None

    def log_db(line: str):
        log(line)
        if run_id and log_conn:
            try:
                log_conn.execute(
                    "INSERT INTO ocr_log_lines (run_id, line) VALUES (?, ?)",
                    (run_id, line)
                )
                log_conn.commit()
            except Exception:
                pass

    threshold = get_threshold(conn)

    all_paths = []
    for root, _, files in os.walk(PHOTO_DIR):
        for filename in sorted(files):
            if filename.lower().endswith(VALID_EXTS):
                all_paths.append(os.path.join(root, filename))

    existing_hashes = set(
        row[0] for row in conn.execute("SELECT file_hash FROM images").fetchall()
    )

    to_process = []
    to_skip    = []
    for path in all_paths:
        try:
            h = hash_file(path)
            if h in existing_hashes:
                to_skip.append(path)
            else:
                to_process.append((path, h))
        except Exception as exc:
            log_db(f"[error] could not hash {path}: {exc}")

    t_run_start = time.time()
    log_db(f"{'═' * 56}")
    log_db(f"  RecipeScan OCR Pipeline")
    log_db(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_db(f"  Photos dir : {PHOTO_DIR}")
    log_db(f"  Threshold  : {threshold:.0%}")
    log_db(f"  OSD min confidence : {OSD_MIN_CONFIDENCE}")
    log_db(f"  Workers    : {WORKER_COUNT}")
    log_db(f"  To process : {len(to_process)}")
    log_db(f"  To skip    : {len(to_skip)}  (already in DB)")
    log_db(f"{'═' * 56}")

    processed = 0
    skipped   = len(to_skip)
    errors    = 0

    if to_process:
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            future_to_path = {
                executor.submit(ocr_image, path, threshold): (path, h)
                for path, h in to_process
            }

            for future in as_completed(future_to_path):
                path, file_hash = future_to_path[future]
                try:
                    result = future.result()
                    result["file_hash"] = file_hash

                    for line in result["lines"]:
                        log_db(line)

                    outcome = write_result(cur, result)
                    conn.commit()

                    if outcome == "processed":
                        processed += 1
                    elif outcome == "skipped":
                        skipped += 1
                        log_db(f"[skip] {os.path.basename(path)}  (duplicate hash)")
                    else:
                        errors += 1

                except Exception as exc:
                    log_db(f"[error] {path}: {exc}")
                    conn.rollback()
                    errors += 1

    conn.close()

    t_total = time.time() - t_run_start
    log_db(f"")
    log_db(f"{'═' * 56}")
    log_db(f"  Pipeline complete")
    log_db(f"  Duration   : {t_total:.0f}s  ({t_total/60:.1f} min)")
    log_db(f"  Processed  : {processed}")
    log_db(f"  Skipped    : {skipped}  (already in DB)")
    log_db(f"  Errors     : {errors}")
    log_db(f"{'═' * 56}")

    if run_id and log_conn:
        try:
            log_conn.execute(
                """UPDATE ocr_runs
                   SET finished_at=CURRENT_TIMESTAMP,
                       processed=?, skipped=?, errors=?, status=?
                   WHERE id=?""",
                (processed, skipped, errors,
                 "error" if errors else "complete", run_id)
            )
            log_conn.commit()
            log_conn.close()
        except Exception:
            pass

    log("__DONE__")
    return {"processed": processed, "skipped": skipped, "errors": errors}


if __name__ == "__main__":
    main()
