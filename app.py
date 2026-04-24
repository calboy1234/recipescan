"""
app.py — RecipeScan Flask Web Server v2

New endpoints over v1
---------------------
POST /admin/run-ocr          — now accepts limit, directory_filter, worker_count, retry_failed
POST /admin/stop-ocr         — gracefully cancels a running pipeline
GET  /admin/progress         — JSON progress for live progress bar polling
POST /admin/retry-errors     — re-queue all previously failed images
GET  /admin/integrity-check  — verify DB consistency and surface any issues
GET  /admin/export           — download results as CSV or JSON
GET  /admin/run/<id>/log     — full log for any historical run
"""

import csv
import gc
import importlib
import io
import json
import mimetypes
import os
import sqlite3
import threading
import time

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, send_file, Response, stream_with_context,
)

from ocr_pipeline import main as run_ocr_pipeline, list_photo_sources
from recipe_detector import extract_title

# ── Configuration ──────────────────────────────────────────────────────────────

app = Flask(__name__)
DB_PATH    = os.environ.get("DB_PATH",    "/data/database/recipescan.db")
PHOTO_DIR  = os.environ.get("PHOTO_DIR",  "/photos")
ITEMS_PER_PAGE = 20

app.jinja_env.filters["fromjson"] = json.loads

# ── Safe filter queries (no f-string interpolation of user input) ──────────────
#
# Each key maps to a complete WHERE clause and its bound parameters.
# filter_val is always validated against this dict before use — unknown values
# fall back to "all" so user-controlled input never reaches the SQL string.

_GALLERY_FILTERS: dict[str, tuple[str, list]] = {
    "all":        ("1=1", []),
    "unreviewed": ("img.is_reviewed = 0 AND ocr.recipe_score >= ?", None),   # threshold injected at call-time
    "detected":   ("ocr.recipe_score >= ?",                          None),
    "reviewed":   ("img.is_reviewed = 1",                            []),
    "unscanned":  ("ocr.id IS NULL",                                 []),
    "errors":     (
        "img.id IN ("
        "  SELECT DISTINCT i2.id FROM failed_images f"
        "  JOIN images i2 ON i2.file_hash = f.file_hash"
        ")",
        [],
    ),
}

# Validated sort options — values are injected directly into ORDER BY so they
# must be hardcoded here; they are never derived from user input.
_SORT_OPTIONS: dict[str, str] = {
    "date_desc":  "img.added_at DESC",
    "date_asc":   "img.added_at ASC",
    "score_desc": "ocr.recipe_score DESC",
    "score_asc":  "ocr.recipe_score ASC",
    "name_asc":   "img.file_path ASC",
    "name_desc":  "img.file_path DESC",
}
_DEFAULT_SORT = "date_desc"

# ── Global pipeline state ──────────────────────────────────────────────────────

ocr_lock       = threading.Lock()
ocr_running    = False
ocr_thread     = None
ocr_stop_event = threading.Event()   # set() to request a graceful stop
ocr_run_id     = None                # ID of the currently active ocr_runs row


# ── Database helpers ───────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_setting(key: str, default):
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return type(default)(row[0]) if row else default
    except Exception:
        return default


def get_threshold():
    return get_setting("recipe_threshold", 0.60)


def _database_file_paths(db_path: str) -> list[str]:
    return [
        db_path,
        f"{db_path}-wal",
        f"{db_path}-shm",
        f"{db_path}-journal",
    ]


def _recreate_database_files():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    for path in _database_file_paths(DB_PATH):
        if not os.path.exists(path):
            continue

        last_error = None
        for _ in range(10):
            try:
                gc.collect()
                os.remove(path)
                last_error = None
                break
            except FileNotFoundError:
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.1)

        if last_error is not None:
            raise RuntimeError(f"Could not remove database file: {path}") from last_error

    import init_db as _init_db
    importlib.reload(_init_db)


def _progress_payload(run: sqlite3.Row) -> dict:
    run_id = run["id"] if "id" in run.keys() else None
    total = run["total"] or 0
    current = min(run["current_count"] or 0, total) if total else 0
    pct = round(max(0.0, min(100.0, 100 * current / total)), 1) if total else 0.0
    return {
        "run_id": run_id,
        "status": run["status"],
        "total": total,
        "current": current,
        "pct": pct,
        "processed": run["processed"] or 0,
        "skipped": run["skipped"] or 0,
        "errors": run["errors"] or 0,
        "running": ocr_running,
    }


# ── Gallery & image management ─────────────────────────────────────────────────

@app.route("/")
def index():
    threshold  = get_threshold()
    page       = request.args.get("page", 1, type=int)
    filter_val = request.args.get("filter", "all")
    sort_val   = request.args.get("sort", _DEFAULT_SORT)
    group_val  = request.args.get("group", "none")

    # Validate both against known safe sets
    if filter_val not in _GALLERY_FILTERS:
        filter_val = "all"
    if sort_val not in _SORT_OPTIONS:
        sort_val = _DEFAULT_SORT

    where_clause, params = _GALLERY_FILTERS[filter_val]
    if params is None:
        params = [threshold]

    # When grouping by session, we must handle things differently because we paginate
    # groups (sessions) rather than individual images.
    if group_val == "session":
        conn = get_db()
        # Fetch ALL images matching the filter (but only necessary columns) to group them
        all_rows = conn.execute(
            "SELECT img.id, img.file_path, img.is_reviewed, img.added_at, img.captured_at,"
            "       ocr.id as ocr_id, ocr.recipe_score "
            "FROM images img "
            "LEFT JOIN ocr_results ocr ON img.id = ocr.image_id "
            "WHERE " + where_clause + " "
            "ORDER BY img.captured_at DESC, img.id DESC",
            params,
        ).fetchall()

        # Clustering logic for session view
        sessions = []
        if all_rows:
            from datetime import datetime
            current_group = {"time": all_rows[0]["captured_at"], "entries": [all_rows[0]]}
            sessions.append(current_group)
            
            last_time = None
            try:
                last_time = datetime.strptime(all_rows[0]["captured_at"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass

            for i in range(1, len(all_rows)):
                row = all_rows[i]
                row_time = None
                try:
                    row_time = datetime.strptime(row["captured_at"], "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
                
                is_new_group = True
                if last_time and row_time:
                    diff = abs((last_time - row_time).total_seconds())
                    if diff < 300: # 5 minutes
                        is_new_group = False
                
                if is_new_group:
                    current_group = {"time": row["captured_at"], "entries": [row]}
                    sessions.append(current_group)
                else:
                    current_group["entries"].append(row)
                
                last_time = row_time

            # Post-process sessions for average score
            for s in sessions:
                scores = [r["recipe_score"] for r in s["entries"] if r["recipe_score"] is not None]
                s["avg_score"] = sum(scores) / len(scores) if scores else 0.0

            # Sort sessions
            if sort_val == "date_desc":
                sessions.sort(key=lambda s: s["time"] or "", reverse=True)
            elif sort_val == "date_asc":
                sessions.sort(key=lambda s: s["time"] or "", reverse=False)
            elif sort_val == "score_desc":
                sessions.sort(key=lambda s: s["avg_score"], reverse=True)
            elif sort_val == "score_asc":
                sessions.sort(key=lambda s: s["avg_score"], reverse=False)

        total = len(sessions)
        total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        page = max(1, min(page, total_pages))
        offset = (page - 1) * ITEMS_PER_PAGE
        
        grouped_rows = sessions[offset : offset + ITEMS_PER_PAGE]
        rows = [] # Not used in template when group_val == 'session'
        conn.close()

    else:
        order_by = _SORT_OPTIONS[sort_val]   # safe: hardcoded strings only
        conn = get_db()

        total = conn.execute(
            "SELECT COUNT(DISTINCT img.id) FROM images img "
            "LEFT JOIN ocr_results ocr ON img.id = ocr.image_id "
            "WHERE " + where_clause,
            params,
        ).fetchone()[0]

        total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        page        = max(1, min(page, total_pages))
        offset      = (page - 1) * ITEMS_PER_PAGE

        rows = conn.execute(
            "SELECT img.id, img.file_path, img.is_reviewed, img.added_at, img.captured_at,"
            "       ocr.id as ocr_id, ocr.recipe_score "
            "FROM images img "
            "LEFT JOIN ocr_results ocr ON img.id = ocr.image_id "
            "WHERE " + where_clause + " "
            "ORDER BY " + order_by + " "
            "LIMIT ? OFFSET ?",
            params + [ITEMS_PER_PAGE, offset],
        ).fetchall()
        grouped_rows = []
        conn.close()

    return render_template(
        "index.html",
        rows=rows, grouped_rows=grouped_rows, total=total, page=page, total_pages=total_pages,
        filter_val=filter_val, sort_val=sort_val, group_val=group_val, threshold=threshold,
    )


@app.route("/image/<int:image_id>")
def image_detail(image_id):
    threshold = get_threshold()
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not img:
        conn.close()
        return render_template("error.html", error="Image not found"), 404

    ocr = conn.execute(
        "SELECT * FROM ocr_results WHERE image_id = ? ORDER BY created_at DESC",
        (image_id,),
    ).fetchall()
    conn.close()

    # Read file stats from disk for the detail panel (not stored in DB)
    file_path = img["file_path"]
    file_stat = {}
    if os.path.exists(file_path):
        try:
            stat = os.stat(file_path)
            file_stat["size_bytes"] = stat.st_size
            size_kb = stat.st_size / 1024
            file_stat["size_human"] = (
                f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"
            )
        except OSError:
            pass
        try:
            from PIL import Image as _PIL
            with _PIL.open(file_path) as probe:
                file_stat["dimensions"] = f"{probe.width} × {probe.height} px"
                file_stat["format"]     = probe.format or "unknown"
                # Derive MIME from the actual format PIL detected, not just the extension
                file_stat["mime_type"]  = (
                    mimetypes.types_map.get("." + probe.format.lower(), "")
                    or mimetypes.guess_type(file_path)[0]
                    or "application/octet-stream"
                )
        except Exception:
            pass

    return render_template(
        "image_detail.html",
        img=img, ocr=ocr, threshold=threshold, file_stat=file_stat,
    )


@app.route("/photo/<int:image_id>")
def serve_photo(image_id):
    conn = get_db()
    img = conn.execute("SELECT file_path FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    if not img:
        return render_template("error.html", error="Photo not found"), 404
    file_path = img["file_path"]

    # Path traversal guard: resolve symlinks and confirm the file lives under PHOTO_DIR
    real_photo_dir = os.path.realpath(PHOTO_DIR)
    real_file_path = os.path.realpath(file_path)
    if not real_file_path.startswith(real_photo_dir + os.sep):
        return render_template("error.html", error="Access denied"), 403

    if not os.path.exists(real_file_path):
        return render_template("error.html", error="Photo file not found on disk"), 404
    mime_type, _ = mimetypes.guess_type(real_file_path)
    return send_file(real_file_path, mimetype=mime_type or "image/jpeg")


@app.route("/image/<int:image_id>/review", methods=["POST"])
def toggle_review(image_id):
    conn = get_db()
    img = conn.execute("SELECT is_reviewed FROM images WHERE id = ?", (image_id,)).fetchone()
    if not img:
        conn.close()
        return jsonify({"error": "Image not found"}), 404
    new_status = 1 - img["is_reviewed"]
    conn.execute("UPDATE images SET is_reviewed = ? WHERE id = ?", (new_status, image_id))
    conn.commit()
    conn.close()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"is_reviewed": new_status})
    return_to = request.form.get("return_to", "index")
    return redirect(
        url_for("index") if return_to == "index"
        else url_for("image_detail", image_id=image_id)
    )


# ── Settings ───────────────────────────────────────────────────────────────────

@app.route("/settings", methods=["GET", "POST"])
def settings():
    # GET: redirect old bookmarks to wherever settings now live
    if request.method == "GET":
        return redirect(url_for("admin"))

    conn = get_db()
    updates = {}

    threshold = request.form.get("recipe_threshold", "0.60")
    try:
        updates["recipe_threshold"] = str(max(0.0, min(1.0, float(threshold))))
    except ValueError:
        updates["recipe_threshold"] = "0.60"

    for key, default in [("worker_count", "2"), ("batch_size", "200"),
                          ("commit_every", "50"), ("log_retention_runs", "10")]:
        val = request.form.get(key, default)
        try:
            updates[key] = str(max(1, int(val)))
        except ValueError:
            updates[key] = default

    for k, v in updates.items():
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, v)
        )
    conn.commit()
    conn.close()

    # return_to lets each form redirect back to the page it came from
    return_to = request.form.get("return_to", "admin")
    return redirect(url_for("index") if return_to == "index" else url_for("admin"))


# ── Admin: dashboard ───────────────────────────────────────────────────────────

@app.route("/admin")
def admin():
    conn = get_db()
    threshold = get_threshold()

    stats = {}
    stats["images"]   = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    stats["ocr"]      = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
    stats["detected"] = conn.execute(
        "SELECT COUNT(*) FROM ocr_results WHERE recipe_score >= ?", (threshold,)
    ).fetchone()[0]
    stats["reviewed"] = conn.execute(
        "SELECT COUNT(*) FROM images WHERE is_reviewed = 1"
    ).fetchone()[0]
    stats["unreviewed_detected"] = conn.execute(
        "SELECT COUNT(*) FROM images i "
        "JOIN ocr_results o ON i.id = o.image_id "
        "WHERE i.is_reviewed = 0 AND o.recipe_score >= ?",
        (threshold,),
    ).fetchone()[0]
    stats["errors"] = conn.execute(
        "SELECT COUNT(*) FROM failed_images WHERE retried = 0"
    ).fetchone()[0]

    # All runs for the history table
    runs = conn.execute(
        "SELECT * FROM ocr_runs ORDER BY started_at DESC LIMIT 20"
    ).fetchall()

    last_run = runs[0] if runs else None
    last_log = []
    if last_run:
        log_rows = conn.execute(
            "SELECT line FROM ocr_log_lines WHERE run_id = ? ORDER BY id ASC",
            (last_run["id"],),
        ).fetchall()
        last_log = [r["line"] for r in log_rows]

    settings_rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    settings_dict = {r["key"]: r["value"] for r in settings_rows}
    conn.close()
    return render_template(
        "admin.html",
        stats=stats, runs=runs, last_run=last_run,
        last_log=last_log, ocr_running=ocr_running,
        ocr_run_id=ocr_run_id,
        settings=settings_dict, threshold=threshold,
    )


@app.route("/admin/stats")
def admin_stats():
    conn = get_db()
    threshold = get_threshold()

    stats = {}
    
    # 1. Image coverage
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    scanned_images = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
    stats["coverage"] = {
        "total": total_images,
        "scanned": scanned_images,
        "unscanned": total_images - scanned_images
    }

    # 2. Recipe Detection
    detected = conn.execute(
        "SELECT COUNT(*) FROM ocr_results WHERE recipe_score >= ?", (threshold,)
    ).fetchone()[0]
    stats["recipes"] = {
        "detected": detected,
        "not_detected": scanned_images - detected
    }

    # 3. Score distribution (Histogram)
    # Group scores into 20 bins: 0.0-0.05, 0.05-0.1, ... 0.95-1.0
    distribution = conn.execute("""
        SELECT CAST(recipe_score * 20 AS INTEGER) as bin, COUNT(*) as count
        FROM ocr_results
        GROUP BY bin
        ORDER BY bin ASC
    """).fetchall()
    
    dist_data = [0] * 20
    for row in distribution:
        bin_idx = min(row["bin"], 19)
        dist_data[bin_idx] += row["count"]
    stats["distribution"] = dist_data

    # 4. Processing History (last 10 runs)
    runs = conn.execute("""
        SELECT id, started_at, processed, errors 
        FROM ocr_runs 
        WHERE status='complete' 
        ORDER BY started_at DESC LIMIT 10
    """).fetchall()
    stats["history"] = [dict(r) for r in reversed(runs)]

    # 5. Review Progress
    reviewed = conn.execute("SELECT COUNT(*) FROM images WHERE is_reviewed = 1").fetchone()[0]
    stats["review"] = {
        "reviewed": reviewed,
        "unreviewed": total_images - reviewed
    }

    conn.close()
    return render_template("stats.html", stats=stats, threshold=threshold)


# ── Admin: list photo sources ─────────────────────────────────────────────────

@app.route("/admin/photo-sources")
def photo_sources():
    """Return the list of subfolders under PHOTO_DIR as selectable sources."""
    return jsonify({"sources": list_photo_sources()})


# ── Admin: start pipeline ──────────────────────────────────────────────────────

@app.route("/admin/run-ocr", methods=["POST"])
def run_ocr():
    global ocr_running, ocr_thread, ocr_stop_event, ocr_run_id

    with ocr_lock:
        if ocr_running:
            return "OCR already running", 409

        # Parse run options from the form
        limit        = request.form.get("limit",      type=int) or None
        worker_count = request.form.get("workers",    type=int) or None
        batch_size   = request.form.get("batch_size", type=int) or None
        retry_failed = bool(request.form.get("retry_failed"))

        # Multi-source selection: list of subfolder names; empty / ["all"] → None (scan everything)
        raw_sources = request.form.getlist("sources")
        sources = [s for s in raw_sources if s and s != "all"] or None

        options = {
            "sources":      sources,
            "limit":        limit,
            "worker_count": worker_count,
            "batch_size":   batch_size,
            "retry_failed": retry_failed,
        }

        ocr_running    = True
        ocr_stop_event = threading.Event()

    def ocr_worker():
        global ocr_running, ocr_run_id
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO ocr_runs (status, options) VALUES ('running', ?)",
                (json.dumps(options),),
            )
            conn.commit()
            run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.close()

            with ocr_lock:
                ocr_run_id = run_id

            run_ocr_pipeline(
                run_id=run_id,
                stop_event=ocr_stop_event,
                limit=limit,
                sources=sources,
                worker_count=worker_count,
                batch_size=batch_size,
                retry_failed=retry_failed,
            )
        finally:
            with ocr_lock:
                ocr_running = False
                ocr_run_id  = None

    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()
    return "", 204


# ── Admin: stop pipeline ───────────────────────────────────────────────────────

@app.route("/admin/stop-ocr", methods=["POST"])
def stop_ocr():
    """Signal the running pipeline to stop after finishing its current batch."""
    global ocr_stop_event
    if not ocr_running:
        return jsonify({"status": "not_running"}), 200
    ocr_stop_event.set()
    return jsonify({"status": "stop_requested"}), 200


# ── Admin: progress (for polling) ─────────────────────────────────────────────

@app.route("/admin/progress")
def ocr_progress():
    """
    Return JSON progress info for the most recent run.
    The admin page polls this every 2 s to show a live progress bar.
    """
    conn = get_db()
    run = conn.execute(
        "SELECT id, status, total, current_count, processed, skipped, errors, "
        "started_at, finished_at "
        "FROM ocr_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if not run:
        return jsonify({"status": "idle"})

    return jsonify(_progress_payload(run))


# ── Admin: SSE log stream ──────────────────────────────────────────────────────

@app.route("/admin/ocr-stream")
def ocr_stream():
    def generate():
        conn = get_db()
        last_run = conn.execute(
            "SELECT id FROM ocr_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        conn.close()

        if not last_run:
            yield "data: [error] No OCR run found\n\n"
            return

        run_id       = last_run["id"]
        last_line_id = 0

        while True:
            conn = get_db()
            rows = conn.execute(
                "SELECT id, line FROM ocr_log_lines "
                "WHERE run_id = ? AND id > ? ORDER BY id ASC",
                (run_id, last_line_id),
            ).fetchall()
            run_row = conn.execute(
                "SELECT id, status, total, current_count, processed, skipped, errors "
                "FROM ocr_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            conn.close()

            for row in rows:
                last_line_id = row["id"]
                yield f"data: {row['line'].replace(chr(10), chr(92) + 'n')}\n\n"

            # Push a progress update on every poll so the UI bar is truly real-time
            if run_row:
                progress_payload = json.dumps(_progress_payload(run_row))
                yield f"data: __PROGRESS__:{progress_payload}\n\n"

                if run_row["status"] not in ("running",):
                    yield "data: __DONE__\n\n"
                    break

            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Admin: historical run log ──────────────────────────────────────────────────

@app.route("/admin/run/<int:run_id>/log")
def run_log(run_id):
    """Return the full log for any historical run as plain text."""
    conn = get_db()
    run = conn.execute("SELECT * FROM ocr_runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        conn.close()
        return "Run not found", 404
    lines = conn.execute(
        "SELECT line FROM ocr_log_lines WHERE run_id = ? ORDER BY id ASC", (run_id,)
    ).fetchall()
    conn.close()
    text = "\n".join(r["line"] for r in lines)
    return Response(text, content_type="text/plain; charset=utf-8")


# ── Admin: retry failed images ────────────────────────────────────────────────

@app.route("/admin/retry-errors", methods=["POST"])
def retry_errors():
    """
    Start an OCR run that processes only previously failed images.
    Shortcut for clicking 'Run OCR' with retry_failed=True.
    """
    global ocr_running, ocr_thread, ocr_stop_event, ocr_run_id

    with ocr_lock:
        if ocr_running:
            return "OCR already running", 409

        conn = get_db()
        n_errors = conn.execute(
            "SELECT COUNT(*) FROM failed_images WHERE retried = 0"
        ).fetchone()[0]
        conn.close()

        if n_errors == 0:
            return redirect(url_for("admin"))

        ocr_running    = True
        ocr_stop_event = threading.Event()

    def ocr_worker():
        global ocr_running, ocr_run_id
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO ocr_runs (status, options) VALUES ('running', ?)",
                (json.dumps({"retry_failed": True}),),
            )
            conn.commit()
            run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.close()

            with ocr_lock:
                ocr_run_id = run_id

            run_ocr_pipeline(
                run_id=run_id,
                stop_event=ocr_stop_event,
                retry_failed=True,
            )
        finally:
            with ocr_lock:
                ocr_running = False
                ocr_run_id  = None

    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()
    return redirect(url_for("admin"))


# ── Admin: integrity check ────────────────────────────────────────────────────

@app.route("/admin/integrity-check")
def integrity_check():
    """
    Run a suite of consistency checks on the database and return a JSON report.
    Useful to run after a large batch to verify nothing was silently dropped.
    """
    conn = get_db()
    issues = []
    info   = {}

    # SQLite built-in integrity check
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    info["sqlite_integrity"] = result
    if result != "ok":
        issues.append(f"SQLite integrity_check returned: {result}")

    # Foreign key violations
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    info["fk_violations"] = len(fk_violations)
    if fk_violations:
        issues.append(f"{len(fk_violations)} foreign key violation(s) found.")

    # Images with no OCR result
    no_ocr = conn.execute(
        "SELECT COUNT(*) FROM images i "
        "LEFT JOIN ocr_results o ON i.id = o.image_id "
        "WHERE o.id IS NULL"
    ).fetchone()[0]
    info["images_without_ocr"] = no_ocr
    if no_ocr:
        issues.append(f"{no_ocr} image(s) in DB have no OCR result.")

    # OCR results with no parent image (orphans)
    orphan_ocr = conn.execute(
        "SELECT COUNT(*) FROM ocr_results o "
        "LEFT JOIN images i ON i.id = o.image_id "
        "WHERE i.id IS NULL"
    ).fetchone()[0]
    info["orphan_ocr_results"] = orphan_ocr
    if orphan_ocr:
        issues.append(f"{orphan_ocr} OCR result(s) reference a non-existent image.")

    # Duplicate hashes (should never happen due to UNIQUE constraint)
    dup_hashes = conn.execute(
        "SELECT file_hash, COUNT(*) as c FROM images "
        "GROUP BY file_hash HAVING c > 1"
    ).fetchall()
    info["duplicate_hashes"] = len(dup_hashes)
    if dup_hashes:
        issues.append(f"{len(dup_hashes)} duplicate file hash(es) found in images table.")

    # Stale 'running' runs (container crashed mid-run)
    stale = conn.execute(
        "SELECT COUNT(*) FROM ocr_runs WHERE status='running'"
    ).fetchone()[0]
    info["stale_running_runs"] = stale
    if stale:
        issues.append(f"{stale} run(s) stuck in 'running' state (container may have crashed).")

    # Unretried failures
    pending_errors = conn.execute(
        "SELECT COUNT(*) FROM failed_images WHERE retried=0"
    ).fetchone()[0]
    info["pending_failed_images"] = pending_errors
    if pending_errors:
        issues.append(
            f"{pending_errors} image(s) failed OCR and have not been retried. "
            "Use 'Retry Errors' to re-process them."
        )

    # General stats
    info["total_images"]      = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    info["total_ocr_results"] = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
    info["total_runs"]        = conn.execute("SELECT COUNT(*) FROM ocr_runs").fetchone()[0]
    info["total_log_lines"]   = conn.execute("SELECT COUNT(*) FROM ocr_log_lines").fetchone()[0]
    conn.close()

    return jsonify({
        "ok":     len(issues) == 0,
        "issues": issues,
        "info":   info,
    })


# ── Admin: export ─────────────────────────────────────────────────────────────

@app.route("/admin/export")
def export():
    """
    Export OCR results as CSV or JSON.

    Query params:
      format    : 'csv' (default) or 'json'
      filter    : 'all' | 'detected' | 'reviewed'
      threshold : override recipe threshold (float)
    """
    fmt       = request.args.get("format", "csv")
    filter_by = request.args.get("filter", "all")
    threshold = request.args.get("threshold", type=float) or get_threshold()

    conn = get_db()

    _export_queries: dict[str, tuple[str, list]] = {
        "all":      ("1=1",                    []),
        "detected": ("o.recipe_score >= ?",    [threshold]),
        "reviewed": ("i.is_reviewed = 1",      []),
    }
    if filter_by not in _export_queries:
        filter_by = "all"
    where, params = _export_queries[filter_by]

    rows = conn.execute(
        "SELECT i.id, i.file_path, i.file_hash, i.is_reviewed, i.added_at,"
        "       o.recipe_score, o.signals, o.text, o.rotation_corrected, "
        "       o.osd_confidence, o.word_confidence, o.psm "
        "FROM images i "
        "LEFT JOIN ocr_results o ON i.id = o.image_id "
        "WHERE " + where + " "
        "ORDER BY i.added_at DESC",
        params,
    ).fetchall()
    conn.close()

    if fmt == "json":
        data = []
        for r in rows:
            entry = dict(r)
            if entry.get("signals"):
                try:
                    entry["signals"] = json.loads(entry["signals"])
                except Exception:
                    pass
            data.append(entry)
        return Response(
            json.dumps(data, indent=2),
            content_type="application/json",
            headers={"Content-Disposition": "attachment; filename=recipescan_export.json"},
        )

    # CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "file_path", "file_hash", "is_reviewed", "added_at",
        "recipe_score", "rotation_corrected", "osd_confidence", "word_confidence", "psm",
        "ingredient_score", "keyword_score", "unit_score", "fraction_score",
    ])
    for r in rows:
        signals = {}
        if r["signals"]:
            try:
                signals = json.loads(r["signals"])
            except Exception:
                pass
        writer.writerow([
            r["id"], r["file_path"], r["file_hash"], r["is_reviewed"], r["added_at"],
            r["recipe_score"], r["rotation_corrected"],
            r["osd_confidence"], r["word_confidence"], r["psm"],
            signals.get("ingredient_score", ""),
            signals.get("keyword_score", ""),
            signals.get("unit_score", ""),
            signals.get("fraction_score", ""),
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        content_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=recipescan_export.csv"},
    )


# ── Admin: reset ───────────────────────────────────────────────────────────────

@app.route("/admin/reset-db", methods=["POST"])
def reset_db():
    if ocr_running:
        return render_template(
            "error.html",
            error="Cannot reset the database while OCR is running. Stop the pipeline first.",
        ), 409

    try:
        _recreate_database_files()
    except RuntimeError as exc:
        return render_template("error.html", error=str(exc)), 500
    return redirect(url_for("admin"))


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Server error"), 500


# ── Context processors ─────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    return {"threshold": get_threshold(), "ocr_running": ocr_running}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
