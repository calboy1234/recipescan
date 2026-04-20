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

Bug fixes (v2.1)
----------------
OMP_NUM_THREADS=1:
  Tesseract uses OpenMP internally. Without capping it, each Python worker
  spawns multiple OS threads, creating N_workers × N_omp_threads threads all
  fighting for CPU time and causing a context-switch death spiral. Capping to
  1 makes Python's ThreadPoolExecutor the sole concurrency layer.

PSM 11 → PSM 3:
  PSM 3 ("fully automatic page segmentation") was selected by automated
  grid-search as the best-performing PSM across 13 real recipe images
  (mean_zscore=0.745, mean_conf=90.0). Previous versions used PSM 11
  ("sparse text"); the search found PSM 3 achieves higher real-word ratio
  and Tesseract confidence on this image set.

Busy-wait → concurrent.futures.wait():
  The old inner loop polled every pending future with f.done() and slept 50ms
  if none were ready — burning CPU constantly. Replaced with wait(FIRST_COMPLETED)
  which blocks in the OS until a future actually finishes.

Optimised preprocessing:
  Automated grid-search over 13 recipe images found that a simple global
  greyscale threshold at level 160 (gray_thresh160) outperforms adaptive
  thresholding on this image set. Recipe card photos have high-contrast
  ink on light paper; the spatial complexity of adaptive thresholding
  amplifies sensor noise rather than helping. cv2 and numpy are no longer
  required by the preprocessing step and have been removed as dependencies.

detect_rotation robustness:
  Explicit guard on osd variable presence before regex matching ensures that
  TesseractError on corrupted images is caught cleanly by the outer handler.

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
import statistics
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from PIL import Image as PILImage, ImageOps
import pytesseract
from pytesseract import Output

# ── OpenMP thread cap ──────────────────────────────────────────────────────────
#
# Tesseract uses OpenMP internally. Without this, each of your N Python workers
# will spin up M OpenMP threads, giving you N×M threads all fighting for CPU
# time. Setting OMP_NUM_THREADS=1 tells Tesseract to use exactly one thread per
# call, making your Python-level ThreadPoolExecutor the sole parallelism layer.
# In practice this is always faster than letting OpenMP and Python fight.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from recipe_detector import score_text

PHOTO_DIR  = os.environ.get("PHOTO_DIR", "/photos")
DB_PATH    = os.environ.get("DB_PATH",   "/data/database/recipescan.db")
VALID_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp")

DEFAULT_THRESHOLD    = 0.60
DEFAULT_WORKERS      = 2
DEFAULT_BATCH_SIZE   = 200
DEFAULT_COMMIT_EVERY = 50
DEFAULT_LOG_KEEP     = 10

OSD_MIN_CONFIDENCE = 2.0


def list_photo_sources() -> list[str]:
    """
    Return a sorted list of immediate subfolder names under PHOTO_DIR.

    Each subfolder is treated as an independent image source that can be
    selected individually in the web UI.  If PHOTO_DIR has no subfolders
    (i.e. images live directly in the root), returns an empty list —
    the caller should then treat the root itself as the single source.
    """
    if not os.path.isdir(PHOTO_DIR):
        return []
    return sorted(
        entry.name
        for entry in os.scandir(PHOTO_DIR)
        if entry.is_dir()
    )
OSD_TIMEOUT = 10   # seconds before OSD is abandoned and rotation skipped
OCR_TIMEOUT = 60   # seconds before an OCR call is killed and the image errors

# ── OCR settings (derived from automated parameter search) ─────────────────────
#
# These values were selected by an exhaustive grid-search test suite that scored
# every (scale, psm, oem, preprocess, invert, preserve_spaces, extra_config)
# combination across 13 real recipe images using a composite metric of
# real-word ratio, Tesseract confidence, garbage ratio, and repeat-char ratio.
#
# Winning config (rank 1, mean_zscore=0.745, mean_conf=90.0, real_word_ratio=0.799):
#   preprocess : gray_thresh160   (grayscale → global threshold at 160)
#   psm        : 3                (fully automatic page segmentation)
#   oem        : 3                (default — LSTM + Legacy)
#   preserve_spaces : True        (-c preserve_interword_spaces=1)
#   extra_config    : -c textord_heavy_nr=1
#   invert     : False  (no inversion)
#   scale      : 1.0   (no rescaling)
#
# Previous pipeline used adaptive thresholding (gray_gaussian_adaptive) and
# PSM 11 (sparse text).  The search found that a simple global threshold at
# grey-level 160 outperforms adaptive thresholding on this image set because
# recipe card photos typically have high-contrast dark ink on light paper;
# the spatial variation that adaptive thresholding is designed to handle is
# less of an issue here than the noise amplification it introduces.
OCR_PSM              = 3     # page segmentation: fully automatic
OCR_OEM              = 3     # OCR engine: LSTM + Legacy (Tesseract default)
OCR_PRESERVE_SPACES  = True  # preserve inter-word spacing in output
OCR_EXTRA_CONFIG     = "-c textord_heavy_nr=1"  # suppress heavy noise regions


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
        osd = pytesseract.image_to_osd(img, config="--psm 0 --dpi 150",
                                       timeout=OSD_TIMEOUT)
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


def preprocess(img: PILImage.Image) -> PILImage.Image:
    """
    Greyscale → global threshold at level 160 (gray_thresh160).

    Selected by automated grid-search as the top-performing preprocessing
    pipeline across 13 recipe images (mean_zscore=0.745, mean_conf=90.0,
    real_word_ratio=0.799).

    Pipeline:
      1. Convert to greyscale (mode "L").
      2. Apply a global point threshold: pixels with value ≥ 160 → 255 (white),
         pixels with value < 160 → 0 (black).

    Why a global threshold beats adaptive here:
      Recipe card photos typically feature high-contrast dark ink on light
      paper.  Adaptive thresholding computes a per-region threshold to handle
      uneven lighting and shadows, but on this image set that extra complexity
      amplifies sensor noise rather than suppressing it.  A single grey-level
      cutoff at 160 is sufficient to cleanly separate ink from background and
      avoids the noise artefacts that hurt adaptive thresholding's scores.

    The output is a pure black-and-white image (0 or 255 per pixel).
    """
    return img.convert("L").point(lambda p: 255 if p >= 160 else 0)


def ocr_with_confidence(img: PILImage.Image, psm: int) -> tuple[str, float]:
    """
    Run Tesseract with the given PSM and return (text, avg_word_confidence).

    Uses image_to_data instead of image_to_string so we get per-word
    confidence scores (0–100) alongside the text.  Rows with conf == -1
    are structural metadata (blocks/paragraphs/lines), not words — filtered
    out before averaging.

    Text is reconstructed by grouping words into lines using the
    (block_num, par_num, line_num) tuple from the data dict, then joining
    lines with newlines.  This preserves the document structure that
    recipe_detector relies on (ingredient lists, keyword density per line)
    rather than collapsing everything into a single space-separated string.
    """
    parts = [f"--oem {OCR_OEM}", f"--psm {psm}"]
    if OCR_PRESERVE_SPACES:
        parts.append("-c preserve_interword_spaces=1")
    if OCR_EXTRA_CONFIG:
        parts.append(OCR_EXTRA_CONFIG)
    tess_config = " ".join(parts)

    data = pytesseract.image_to_data(
        img,
        config=tess_config,
        output_type=Output.DICT,
        timeout=OCR_TIMEOUT,
    )

    line_words: dict[tuple, list[str]] = {}
    confidences: list[int] = []

    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf == -1:          # structural metadata row, not a word
            continue
        if not word.strip():    # empty string between words
            continue
        confidences.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_words.setdefault(key, []).append(word)

    text     = "\n".join(" ".join(words) for words in line_words.values())
    avg_conf = statistics.mean(confidences) if confidences else 0.0
    return text, round(avg_conf, 2)


def run_ocr(path: str, lines: list) -> tuple[str, int, float, float]:
    """
    Open, orient, preprocess, then OCR the image with PSM 3.

    Returns (text, angle_corrected, osd_confidence, word_confidence).

    PSM strategy
    ------------
    PSM 3 ("fully automatic page segmentation") is used throughout.  It
    was selected by automated grid-search as the best-performing PSM for
    this image set (mean_zscore=0.745).  PSM 3 lets Tesseract detect the
    page orientation and script before segmenting, which works well for
    recipe images that typically have a clear dominant text block (even if
    that block is a multi-column ingredient list).

    Memory management
    -----------------
    PIL images are closed as soon as they are no longer needed so that file
    handles and pixel buffers are released promptly across long runs.
    img.load() inside the 'with' block forces pixel data into memory before
    the file handle closes, making the subsequent objects self-contained.
    """
    with PILImage.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        img.load()          # force pixel data into memory before file closes
    # 'raw' file handle is now closed; 'img' owns its pixel data

    # detect_rotation is run on the grayscale version of the image.
    # Orientation detection is more reliable on grayscale than on binarized
    # images, where important anti-aliasing details are lost.
    grey = img.convert("L")

    osd_conf: float
    angle, osd_conf = detect_rotation(grey)

    if angle != 0:
        rotated = grey.rotate(angle, expand=True)
        grey.close()
        grey = rotated
        lines.append(
            f"  rotation  : {angle}° corrected  "
            f"(OSD confidence: {osd_conf:.2f})"
        )
    elif osd_conf > 0:
        label = (
            "below threshold, skipped"
            if osd_conf < OSD_MIN_CONFIDENCE
            else "already upright"
        )
        lines.append(f"  rotation  : none  (OSD confidence: {osd_conf:.2f} — {label})")
    else:
        lines.append("  rotation  : OSD unavailable or image has too little text")

    # Final preprocessing (binarization) happens AFTER rotation.
    binarized = preprocess(grey)
    grey.close()
    img.close()

    text, word_conf = ocr_with_confidence(binarized, psm=OCR_PSM)
    binarized.close()

    return text, angle, osd_conf, word_conf


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
        "osd_confidence": 0.0, "word_confidence": 0.0,
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
        text, angle, osd_confidence, word_confidence = run_ocr(path, lines)
        result["ocr_time"]        = time.time() - t0
        result["text"]            = text
        result["angle"]           = angle
        result["osd_confidence"]  = osd_confidence
        result["word_confidence"] = word_confidence

        lines.append(f"  ocr time  : {result['ocr_time']:.1f}s  (word confidence: {word_confidence:.0f}%)")

        detection = score_text(text)
        result["recipe_score"] = detection["score"]
        result["signals"]      = detection["signals"]
        sig = detection["signals"]

        lines.append(
            f"  score     : {detection['score']:.0%}  "
            f"(keywords={sig['keyword_score']:.2f}  "
            f"units={sig['unit_score']:.2f}  "
            f"fractions={sig['fraction_score']:.2f}  "
            f"ingredients={sig['ingredient_score']:.2f})"
        )

        label = "RECIPE DETECTED — review in RecipeScan UI" if detection["score"] >= threshold else "not a recipe"
        lines.append(f"  result    : {label}")
        lines.append(f"└─ done in {result['ocr_time']:.1f}s")
        result["status"] = "ok"

    except pytesseract.TesseractError as exc:
        msg = str(exc)
        result["error_msg"] = msg
        if "Tesseract run timed out" in msg or "TimeoutExpired" in msg:
            lines.append(f"  [timeout] : Tesseract did not finish within the allowed time")
            lines.append(f"  [timeout] : image will be recorded as a failure for later retry")
        else:
            lines.append(f"  [error]   : tesseract — {exc}")
        lines.append(f"└─ failed")

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
            (image_id, engine, text, osd_confidence, word_confidence,
             psm, recipe_score, signals, rotation_corrected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id, "tesseract", result["text"],
            round(result.get("osd_confidence", 0.0), 2),
            result.get("word_confidence", 0.0),
            OCR_PSM,
            result["recipe_score"], json.dumps(result["signals"]),
            result["angle"],
        ),
    )
    return "processed"


def write_failure(cur: sqlite3.Cursor, run_id: Optional[int], result: dict):
    """Record a failed image in failed_images for later retry."""
    # When running from the CLI (run_id=None), we still want to record 
    # failures if possible, but the foreign key constraint on run_id 
    # must be respected. We only insert if we have a valid run_id.
    if run_id is None:
        return

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
    sources: list = None,
    directory_filter: str = None,   # deprecated alias → wrapped into sources
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
    sources          : list of subfolder names under PHOTO_DIR to scan.
                       None or [] = scan all of PHOTO_DIR.
    directory_filter : deprecated single-folder alias; still accepted so
                       existing callers don't break.
    retry_failed     : if True, re-process images from failed_images instead
                       of the normal new-image scan
    worker_count     : override the worker_count setting
    batch_size       : override the batch_size setting
    """
    if stop_event is None:
        stop_event = threading.Event()

    # Back-compat: wrap legacy directory_filter into sources list
    if directory_filter and not sources:
        sources = [directory_filter]

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
    #
    # Uses the same connection (conn) as all other DB writes rather than a
    # separate log_conn.  A second connection creates write-lock contention:
    # conn holds an open uncommitted write transaction across commit_every
    # images, and WAL mode only allows one writer at a time.  log_conn would
    # block for up to the sqlite3 busy-timeout (5 s default) on every INSERT,
    # stalling the main thread and preventing it from collecting completed
    # futures via wait().  A single connection eliminates the contention.
    #
    # Log lines are NOT committed inside log_db — they are flushed by the
    # explicit conn.commit() that immediately follows each image's log block
    # in the main loop below, giving the web UI near-real-time updates.

    def log_db(line: str):
        log(line)
        if run_id:
            try:
                conn.execute(
                    "INSERT INTO ocr_log_lines (run_id, line) VALUES (?, ?)",
                    (run_id, line)
                )
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
        # We no longer mark as retried here; it's done after processing each image 
        # in the loop below to ensure crashed runs don't lose images.
        log_db("  Mode       : RETRY FAILED IMAGES")
    else:
        # Normal scan: collect image paths from one, some, or all sources.
        # If `sources` is None/empty → scan the whole PHOTO_DIR root.
        # Otherwise → scan each named subfolder and union the results.
        if sources:
            scan_roots = []
            for src in sources:
                candidate = os.path.join(PHOTO_DIR, src.lstrip("/"))
                if os.path.isdir(candidate):
                    scan_roots.append(candidate)
                else:
                    log_db(f"WARNING: source '{src}' not found under {PHOTO_DIR}, skipping.")
            if not scan_roots:
                log_db("ERROR: none of the requested sources exist — falling back to full scan.")
                scan_roots = [PHOTO_DIR]
        else:
            scan_roots = [PHOTO_DIR]

        all_paths = []
        for scan_root in scan_roots:
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
    if sources:
        log_db(f"  Sources    : {', '.join(sources)}")
    else:
        log_db(f"  Sources    : all")
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
            # Block until at least one future completes (or 0.5 s timeout to
            # allow the stop_event check to fire). This replaces the old
            # busy-wait poll loop and lets the OS do the blocking efficiently.
            done_futures, _ = wait(
                pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED
            )

            # Respect stop requests — checked after every wait() cycle
            if stop_event.is_set():
                log_db("")
                log_db("[stopped] Stop requested — draining current batch then exiting.")
                for f in list(pending):
                    f.cancel()
                break

            if not done_futures:
                continue

            for future in done_futures:
                path, file_hash = pending.pop(future)
                try:
                    result = future.result()
                    result["file_hash"] = file_hash  # ensure pre-computed hash is used

                    for line in result["lines"]:
                        log_db(line)
                    # Flush log lines to the DB immediately so the web UI
                    # SSE stream can read them without waiting for the next
                    # commit_every boundary.  This commit is cheap (log-only
                    # rows) and replaces the old log_conn-per-line commits.
                    conn.commit()

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
                        write_failure(cur, run_id, result)

                    # Mark as retried in failed_images regardless of success/failure
                    cur.execute(
                        "UPDATE failed_images SET retried = 1 WHERE file_path = ? AND retried = 0",
                        (path,)
                    )

                except Exception as exc:
                    log_db(f"[error] {path}: {exc}")
                    errors += 1
                    write_failure(cur, run_id, {
                        "path": path,
                        "file_hash": file_hash,
                        "error_msg": str(exc)
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
