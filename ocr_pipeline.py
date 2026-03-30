"""
ocr_pipeline.py — RecipeScan OCR Pipeline v2

Designed to handle hundreds of thousands of images reliably.

Key improvements over v1
------------------------
Bounded memory:
  Jobs are submitted in chunks of `batch_size` (default 200). The old code
  submitted all futures at once, which would create hundreds of thousands of
  Future objects in memory simultaneously.

Stop/cancel:
  Pass a threading.Event as `stop_event`. The pipeline drains the current
  chunk and exits cleanly, writing status='stopped' to the run record.

Progress tracking:
  `ocr_runs.total` and `ocr_runs.current_count` are updated every
  `commit_every` completions so the web UI can show a live progress bar
  without relying solely on log-line streaming.

Failure tracking:
  Every image that raises an exception is recorded in `failed_images`.
  Pass `retry_failed=True` to re-process only those images (bypassing the
  normal hash-based skip so they get a fresh attempt).

No double-hashing:
  The pre-run scan hashes each file once. That hash is passed into the worker
  so OCR threads never recompute it.

Scope controls:
  `limit`            — process at most N images (useful for staged roll-outs)
  `directory_filter` — restrict to images under a specific subdirectory
  `retry_failed`     — process only images that failed in previous runs
  `worker_count`     — override the setting from the DB
  `batch_size`       — override the setting from the DB

Log pruning:
  After each completed run, log lines for runs older than `log_retention_runs`
  are deleted so the log table does not grow without bound.

Parallelism model
-----------------
  ThreadPoolExecutor with configurable workers. DB writes are serialised in
  the calling thread (main thread or web worker thread). Workers do only
  CPU/IO work (hashing, PIL, Tesseract) and return plain dicts.
"""

import json
import os
import re
import time
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

DEFAULT_THRESHOLD    = 0.75
DEFAULT_WORKERS      = 4
DEFAULT_BATCH_SIZE   = 200
DEFAULT_COMMIT_EVERY = 50
DEFAULT_LOG_KEEP     = 10

OSD_MIN_CONFIDENCE = 2.0


# ── Settings helpers ───────────────────────────────────────────────────────────

def _get_setting(conn: sqlite3.Connection, key: str, default):
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        val = row[0]
        return type(default)(val)
    except Exception:
        return default


# ── Utilities ──────────────────────────────────────────────────────────────────

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
    return f"{b/1024/1024:.1f}MB"


# ── Orientation / OCR ──────────────────────────────────────────────────────────

def detect_rotation(img: PILImage.Image) -> tuple[int, float]:
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
        label = (
            "below threshold, skipped"
            if confidence < OSD_MIN_CONFIDENCE
            else "already upright"
        )
        lines.append(f"  rotation  : none  (OSD confidence: {confidence:.2f} — {label})")
    else:
        lines.append("  rotation  : OSD unavailable or image has too little text")

    text = pytesseract.image_to_string(grey, config=r"--psm 6")
    return text, angle, confidence


# ── Worker (runs in thread pool, no DB access) ─────────────────────────────────

def ocr_image(path: str, file_hash: str, threshold: float) -> dict:
    """
    Process one image. Returns a result dict; the caller handles all DB writes.
    The pre-computed `file_hash` is accepted directly — no double hashing.
    """
    lines  = []
    result = {
        "path": path, "status": "error", "lines": lines,
        "file_hash": file_hash, "text": "", "angle": 0,
        "recipe_score": 0.0, "signals": {},
        "file_size": 0, "dimensions": "unknown",
        "ocr_time": 0.0, "error_msg": "",
    }

    try:
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

        label = "RECIPE DETECTED — review in RecipeScan UI" if detection["score"] >= threshold else "not a recipe"
        lines.append(f"  result    : {label}")
        lines.append(f"└─ done in {result['ocr_time']:.1f}s")
        result["status"] = "ok"

    except Exception as exc:
        result["error_msg"] = str(exc)
        lines.append(f"  [error]   : {exc}")
        lines.append(f"└─ failed")

    return result


# ── DB writes (caller's thread only) ──────────────────────────────────────────

def write_result(cur: sqlite3.Cursor, result: dict) -> str:
    """Write a completed OCR result. Returns 'processed', 'skipped', or 'error'."""
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


def write_failure(cur: sqlite3.Cursor, run_id: int, result: dict):
    """Record a failed image in failed_images for later retry."""
    cur.execute(
        """
        INSERT INTO failed_images (run_id, file_path, file_hash, error_msg)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, result["path"], result.get("file_hash"), result.get("error_msg", "")),
    )


def prune_old_logs(conn: sqlite3.Connection, keep_runs: int):
    """Delete log lines for all but the most recent `keep_runs` runs."""
    try:
        ids = conn.execute(
            "SELECT id FROM ocr_runs ORDER BY started_at DESC LIMIT ?",
            (keep_runs,)
        ).fetchall()
        if not ids:
            return
        keep_ids = [str(r[0]) for r in ids]
        placeholders = ",".join(keep_ids)
        conn.execute(
            f"DELETE FROM ocr_log_lines WHERE run_id NOT IN ({placeholders})"
        )
        conn.commit()
    except Exception:
        pass


# ── Main entry point ───────────────────────────────────────────────────────────

def main(
    log=print,
    run_id: int = None,
    stop_event: threading.Event = None,
    *,
    limit: int = None,
    directory_filter: str = None,
    retry_failed: bool = False,
    worker_count: int = None,
    batch_size: int = None,
) -> dict:
    """
    Run the full OCR pipeline.

    Parameters
    ----------
    log              : callable receiving a single string (default: print)
    run_id           : ocr_runs row ID for persisting log lines and progress
    stop_event       : threading.Event; set it to request a graceful stop
    limit            : process at most this many images (None = no limit)
    directory_filter : only process images under this subdirectory of PHOTO_DIR
    retry_failed     : if True, re-process images from failed_images instead
                       of the normal new-image scan
    worker_count     : override the worker_count setting
    batch_size       : override the batch_size setting
    """
    if stop_event is None:
        stop_event = threading.Event()

    if not os.path.isdir(PHOTO_DIR):
        log("ERROR: /photos is not mounted or does not exist.")
        return {"processed": 0, "skipped": 0, "errors": 0}

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # ── Load settings ──────────────────────────────────────────────────────────
    threshold    = _get_setting(conn, "recipe_threshold",   DEFAULT_THRESHOLD)
    n_workers    = worker_count  or _get_setting(conn, "worker_count",      DEFAULT_WORKERS)
    n_batch      = batch_size    or _get_setting(conn, "batch_size",        DEFAULT_BATCH_SIZE)
    commit_every = _get_setting(conn, "commit_every",    DEFAULT_COMMIT_EVERY)
    log_keep     = _get_setting(conn, "log_retention_runs", DEFAULT_LOG_KEEP)

    # ── Log helper (persists lines if run_id given) ────────────────────────────
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

    # ── Progress update helper ─────────────────────────────────────────────────
    def update_progress(current: int, total: int):
        if run_id:
            try:
                conn.execute(
                    "UPDATE ocr_runs SET current_count=?, total=? WHERE id=?",
                    (current, total, run_id)
                )
                # Note: caller commits as part of normal commit cycle
            except Exception:
                pass

    # ── Build the work list ────────────────────────────────────────────────────

    if retry_failed:
        # Re-process images that failed in any previous run.
        # We delete their existing failed_images rows so a fresh failure gets
        # recorded cleanly. We do NOT skip by hash — that's the whole point.
        rows = conn.execute(
            "SELECT DISTINCT file_path, file_hash FROM failed_images WHERE retried = 0"
        ).fetchall()
        to_process = [(r[0], r[1]) for r in rows if r[1]]
        # Mark them as retried so they aren't re-queued by future retry runs
        conn.execute("UPDATE failed_images SET retried=1 WHERE retried=0")
        conn.commit()
        log_db("  Mode       : RETRY FAILED IMAGES")
    else:
        # Normal scan: walk PHOTO_DIR and skip anything already in the DB.
        scan_root = PHOTO_DIR
        if directory_filter:
            candidate = os.path.join(PHOTO_DIR, directory_filter.lstrip("/"))
            if os.path.isdir(candidate):
                scan_root = candidate
            else:
                log_db(f"WARNING: directory_filter '{directory_filter}' not found, scanning all.")

        all_paths = []
        for root, _, files in os.walk(scan_root):
            for filename in sorted(files):
                if filename.lower().endswith(VALID_EXTS):
                    all_paths.append(os.path.join(root, filename))

        existing_hashes = set(
            row[0] for row in conn.execute("SELECT file_hash FROM images").fetchall()
        )

        to_process = []
        skipped_pre = 0
        for path in all_paths:
            try:
                h = hash_file(path)
                if h in existing_hashes:
                    skipped_pre += 1
                else:
                    to_process.append((path, h))
            except Exception as exc:
                log_db(f"[error] could not hash {path}: {exc}")

    # Apply limit
    if limit and len(to_process) > limit:
        to_process = to_process[:limit]
        log_db(f"  Limit      : capped at {limit} images")

    total = len(to_process)

    # ── Header ─────────────────────────────────────────────────────────────────

    t_run_start = time.time()
    log_db(f"{'═' * 56}")
    log_db(f"  RecipeScan OCR Pipeline v2")
    log_db(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_db(f"  Photos dir : {PHOTO_DIR}")
    log_db(f"  Threshold  : {threshold:.0%}")
    log_db(f"  Workers    : {n_workers}")
    log_db(f"  Batch size : {n_batch}")
    log_db(f"  Commit every : {commit_every} images")
    if not retry_failed:
        log_db(f"  Pre-skipped: {skipped_pre}  (already in DB)")
    log_db(f"  To process : {total}")
    log_db(f"{'═' * 56}")

    # Write total to run record so the progress bar has a denominator
    if run_id:
        conn.execute("UPDATE ocr_runs SET total=? WHERE id=?", (total, run_id))
        conn.commit()

    processed = 0
    skipped   = skipped_pre if not retry_failed else 0
    errors    = 0
    commit_buf = 0   # images written since last commit

    # ── Process in bounded chunks ──────────────────────────────────────────────
    #
    # Submitting all futures at once for 500k images would create 500k Future
    # objects in memory. Instead we slide a window of `n_batch` futures,
    # always keeping at most n_batch in flight. This bounds peak memory.

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        paths_iter = iter(to_process)
        pending    = {}

        def _submit_next():
            item = next(paths_iter, None)
            if item is None:
                return
            path, h = item
            f = executor.submit(ocr_image, path, h, threshold)
            pending[f] = (path, h)

        # Fill initial window
        for _ in range(min(n_batch, total)):
            _submit_next()

        while pending:
            # Respect stop requests between chunks
            if stop_event.is_set():
                log_db("")
                log_db("[stopped] Stop requested — draining current batch then exiting.")
                # Cancel pending futures if possible (best-effort)
                for f in list(pending):
                    f.cancel()
                break

            # Wait for the next completion
            done_futures = []
            for f in list(pending):
                if f.done():
                    done_futures.append(f)

            if not done_futures:
                # Nothing done yet — wait briefly then retry
                import time as _time
                _time.sleep(0.05)
                continue

            for future in done_futures:
                path, file_hash = pending.pop(future)
                try:
                    result = future.result()
                    result["file_hash"] = file_hash  # ensure pre-computed hash is used

                    for line in result["lines"]:
                        log_db(line)

                    outcome = write_result(cur, result)

                    if outcome == "processed":
                        processed += 1
                    elif outcome == "skipped":
                        skipped += 1
                        log_db(f"[skip] {os.path.basename(path)}  (duplicate hash)")
                    else:  # error from write_result (shouldn't happen for status=ok)
                        errors += 1

                    if result["status"] == "error":
                        errors += 1
                        write_failure(cur, run_id or 0, result)

                except Exception as exc:
                    log_db(f"[error] {path}: {exc}")
                    errors += 1
                    write_failure(cur, run_id or 0, {
                        "path": path,
                        "file_hash": file_hash,
                        "error_msg": str(exc),
                    })

                # Commit on schedule
                commit_buf += 1
                if commit_buf >= commit_every:
                    current_done = processed + skipped + errors
                    update_progress(current_done, total)
                    conn.commit()
                    commit_buf = 0

                # Slide the window: submit one more to replace the completed one
                _submit_next()

    # Final commit for anything still buffered
    current_done = processed + skipped + errors
    update_progress(current_done, total)
    conn.commit()

    t_total = time.time() - t_run_start

    # ── Summary ────────────────────────────────────────────────────────────────

    log_db("")
    log_db(f"{'═' * 56}")
    log_db(f"  Pipeline {'stopped early' if stop_event.is_set() else 'complete'}")
    log_db(f"  Duration   : {t_total:.0f}s  ({t_total/60:.1f} min)")
    log_db(f"  Processed  : {processed}")
    log_db(f"  Skipped    : {skipped}  (already in DB)")
    log_db(f"  Errors     : {errors}")
    if errors:
        log_db(f"  → Use 'Retry Errors' in the admin UI to re-process failed images.")
    log_db(f"{'═' * 56}")

    # ── Finalise run record ────────────────────────────────────────────────────

    if run_id:
        final_status = (
            "stopped"   if stop_event.is_set()
            else "error" if errors and processed == 0
            else "complete"
        )
        conn.execute(
            """UPDATE ocr_runs
               SET finished_at=CURRENT_TIMESTAMP,
                   processed=?, skipped=?, errors=?,
                   total=?, current_count=?,
                   status=?
               WHERE id=?""",
            (processed, skipped, errors,
             total, current_done,
             final_status, run_id)
        )
        conn.commit()

    conn.close()

    # ── Prune old log lines ────────────────────────────────────────────────────
    try:
        prune_conn = sqlite3.connect(DB_PATH)
        prune_old_logs(prune_conn, log_keep)
        prune_conn.close()
    except Exception:
        pass

    if log_conn:
        try:
            log_conn.close()
        except Exception:
            pass

    log("__DONE__")
    return {"processed": processed, "skipped": skipped, "errors": errors}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecipeScan OCR Pipeline")
    parser.add_argument("--limit",    type=int,  default=None, help="Max images to process")
    parser.add_argument("--dir",      type=str,  default=None, help="Subdirectory filter")
    parser.add_argument("--workers",  type=int,  default=None, help="Worker thread count")
    parser.add_argument("--batch",    type=int,  default=None, help="Batch size")
    parser.add_argument("--retry",    action="store_true",     help="Retry failed images")
    args = parser.parse_args()

    main(
        limit=args.limit,
        directory_filter=args.dir,
        worker_count=args.workers,
        batch_size=args.batch,
        retry_failed=args.retry,
    )
