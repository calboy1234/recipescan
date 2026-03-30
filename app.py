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

from ocr_pipeline import main as run_ocr_pipeline
from recipe_detector import extract_title

# ── Configuration ──────────────────────────────────────────────────────────────

app = Flask(__name__)
DB_PATH = os.environ.get("DB_PATH", "/data/database/recipescan.db")
ITEMS_PER_PAGE = 20

app.jinja_env.filters["fromjson"] = json.loads

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
    return get_setting("recipe_threshold", 0.75)


# ── Gallery & image management ─────────────────────────────────────────────────

@app.route("/")
def index():
    threshold  = get_threshold()
    page       = request.args.get("page", 1, type=int)
    filter_val = request.args.get("filter", "all")

    conn = get_db()
    where_clauses = []
    params = []

    if filter_val == "unreviewed":
        where_clauses.append("img.is_reviewed = 0 AND ocr.recipe_score >= ?")
        params = [threshold]
    elif filter_val == "detected":
        where_clauses.append("ocr.recipe_score >= ?")
        params = [threshold]
    elif filter_val == "reviewed":
        where_clauses.append("img.is_reviewed = 1")
    elif filter_val == "unscanned":
        where_clauses.append("ocr.id IS NULL")
    elif filter_val == "errors":
        where_clauses.append(
            "img.id IN (SELECT DISTINCT f.file_hash "
            "FROM failed_images f "
            "JOIN images i2 ON i2.file_hash = f.file_hash)"
        )

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    total = conn.execute(
        f"SELECT COUNT(*) FROM images img "
        f"LEFT JOIN ocr_results ocr ON img.id = ocr.image_id "
        f"WHERE {where_clause}",
        params,
    ).fetchone()[0]

    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page        = max(1, min(page, total_pages))
    offset      = (page - 1) * ITEMS_PER_PAGE

    rows = conn.execute(
        f"""
        SELECT img.id, img.file_path, img.is_reviewed, img.added_at,
               ocr.id as ocr_id, ocr.recipe_score
        FROM images img
        LEFT JOIN ocr_results ocr ON img.id = ocr.image_id
        WHERE {where_clause}
        ORDER BY img.added_at DESC
        LIMIT ? OFFSET ?
        """,
        params + [ITEMS_PER_PAGE, offset],
    ).fetchall()

    conn.close()
    return render_template(
        "index.html",
        rows=rows, total=total, page=page, total_pages=total_pages,
        filter_val=filter_val, threshold=threshold,
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
    return render_template("image_detail.html", img=img, ocr=ocr, threshold=threshold)


@app.route("/photo/<int:image_id>")
def serve_photo(image_id):
    conn = get_db()
    img = conn.execute("SELECT file_path FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    if not img:
        return render_template("error.html", error="Photo not found"), 404
    file_path = img["file_path"]
    if not os.path.exists(file_path):
        return render_template("error.html", error="Photo file not found on disk"), 404
    mime_type, _ = mimetypes.guess_type(file_path)
    return send_file(file_path, mimetype=mime_type or "image/jpeg")


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
    conn = get_db()
    if request.method == "POST":
        updates = {}

        threshold = request.form.get("recipe_threshold", "0.75")
        try:
            updates["recipe_threshold"] = str(max(0.0, min(1.0, float(threshold))))
        except ValueError:
            updates["recipe_threshold"] = "0.75"

        for key, default in [("worker_count", "4"), ("batch_size", "200"),
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
        return redirect(url_for("settings"))

    rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    conn.close()
    settings_dict = {r["key"]: r["value"] for r in rows}
    return render_template("settings.html", settings=settings_dict)


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

    conn.close()
    return render_template(
        "admin.html",
        stats=stats, runs=runs, last_run=last_run,
        last_log=last_log, ocr_running=ocr_running,
        ocr_run_id=ocr_run_id,
    )


# ── Admin: start pipeline ──────────────────────────────────────────────────────

@app.route("/admin/run-ocr", methods=["POST"])
def run_ocr():
    global ocr_running, ocr_thread, ocr_stop_event, ocr_run_id

    with ocr_lock:
        if ocr_running:
            return "OCR already running", 409

        # Parse run options from the form
        limit            = request.form.get("limit",      type=int) or None
        directory_filter = request.form.get("dir_filter", "").strip() or None
        worker_count     = request.form.get("workers",    type=int) or None
        batch_size       = request.form.get("batch_size", type=int) or None
        retry_failed     = bool(request.form.get("retry_failed"))

        options = {
            "limit":            limit,
            "directory_filter": directory_filter,
            "worker_count":     worker_count,
            "batch_size":       batch_size,
            "retry_failed":     retry_failed,
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
                directory_filter=directory_filter,
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

    total   = run["total"] or 0
    current = run["current_count"] or 0
    pct     = round(100 * current / total, 1) if total else 0

    return jsonify({
        "run_id":    run["id"],
        "status":    run["status"],
        "total":     total,
        "current":   current,
        "pct":       pct,
        "processed": run["processed"],
        "skipped":   run["skipped"],
        "errors":    run["errors"],
        "running":   ocr_running,
    })


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
            run_status = conn.execute(
                "SELECT status FROM ocr_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.close()

            for row in rows:
                last_line_id = row["id"]
                yield f"data: {row['line'].replace(chr(10), chr(92) + 'n')}\n\n"

            if run_status and run_status["status"] not in ("running",):
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

    where = "1=1"
    params: list = []
    if filter_by == "detected":
        where = "o.recipe_score >= ?"
        params = [threshold]
    elif filter_by == "reviewed":
        where = "i.is_reviewed = 1"

    rows = conn.execute(
        f"""
        SELECT i.id, i.file_path, i.file_hash, i.is_reviewed, i.added_at,
               o.recipe_score, o.signals, o.text, o.rotation_corrected
        FROM images i
        LEFT JOIN ocr_results o ON i.id = o.image_id
        WHERE {where}
        ORDER BY i.added_at DESC
        """,
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
        "recipe_score", "rotation_corrected",
        "keyword_score", "unit_score", "fraction_score", "list_score",
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
            signals.get("keyword_score", ""),
            signals.get("unit_score", ""),
            signals.get("fraction_score", ""),
            signals.get("list_score", ""),
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        content_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=recipescan_export.csv"},
    )


# ── Admin: clear / reset ───────────────────────────────────────────────────────

@app.route("/admin/clear-scans", methods=["POST"])
def clear_scans():
    conn = get_db()
    conn.execute("DELETE FROM failed_images")
    conn.execute("DELETE FROM ocr_log_lines")
    conn.execute("DELETE FROM ocr_runs")
    conn.execute("DELETE FROM ocr_results")
    conn.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    return redirect(url_for("admin"))


@app.route("/admin/reset-db", methods=["POST"])
def reset_db():
    conn = get_db()
    for tbl in ("failed_images", "ocr_log_lines", "ocr_runs", "ocr_results", "images", "settings"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()
    import init_db as _init_db
    importlib.reload(_init_db)
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
