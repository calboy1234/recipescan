"""
app.py — RecipeScan Flask Web Server

A web UI for browsing scanned recipe images, reviewing OCR results,
managing pipeline runs, and adjusting detection thresholds.
"""

import os
import json
import sqlite3
import threading
import mimetypes
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.security import safe_str

from ocr_pipeline import main as run_ocr_pipeline
from recipe_detector import extract_title

# ── Configuration ──────────────────────────────────────────────────────────

app = Flask(__name__)
DB_PATH = "/data/database/recipescan.db"
PHOTO_DIR = "/photos"
ITEMS_PER_PAGE = 20

# Global state for OCR pipeline
ocr_lock = threading.Lock()
ocr_running = False
ocr_thread = None

# ── Database Helpers ──────────────────────────────────────────────────────

def get_db():
    """Get a database connection (one per request)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn

def get_threshold():
    """Fetch the recipe detection threshold from settings."""
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'recipe_threshold'"
        ).fetchone()
        conn.close()
        return float(row[0]) if row else 0.75
    except Exception:
        return 0.75

def get_image_status(image_row, threshold):
    """
    Determine the status of an image (unscanned, not_recipe, recipe, etc).
    Returns a tuple: (status_key, badge_label, badge_color)
    """
    if image_row['ocr_id'] is None:
        return ('unscanned', 'Unscanned', 'grey')
    score = image_row['recipe_score'] or 0
    if score >= threshold:
        return ('detected', f'Recipe ({score*100:.0f}%)', 'orange')
    else:
        return ('not_recipe', 'Not a recipe', 'grey')

# ── Routes: Gallery & Image Management ─────────────────────────────────

@app.route('/')
def index():
    """Image gallery with filtering and pagination."""
    threshold = get_threshold()
    page = request.args.get('page', 1, type=int)
    filter_val = request.args.get('filter', 'all')
    
    conn = get_db()
    
    # Build WHERE clause based on filter
    where_clauses = []
    if filter_val == 'unreviewed':
        where_clauses.append("img.is_reviewed = 0 AND ocr.recipe_score >= ?")
        params = [threshold]
    elif filter_val == 'detected':
        where_clauses.append("ocr.recipe_score >= ?")
        params = [threshold]
    elif filter_val == 'reviewed':
        where_clauses.append("img.is_reviewed = 1")
        params = []
    elif filter_val == 'unscanned':
        where_clauses.append("ocr.id IS NULL")
        params = []
    else:  # 'all'
        params = []
    
    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    # Count total for this filter
    query_count = f"""
        SELECT COUNT(*) as cnt FROM images img
        LEFT JOIN ocr_results ocr ON img.id = ocr.image_id
        WHERE {where_clause}
    """
    total = conn.execute(query_count, params).fetchone()['cnt']
    total_pages = (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    
    # Fetch paginated results
    offset = (page - 1) * ITEMS_PER_PAGE
    query = f"""
        SELECT 
            img.id,
            img.file_path,
            img.is_reviewed,
            img.added_at,
            ocr.id as ocr_id,
            ocr.recipe_score
        FROM images img
        LEFT JOIN ocr_results ocr ON img.id = ocr.image_id
        WHERE {where_clause}
        ORDER BY img.added_at DESC
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(query, params + [ITEMS_PER_PAGE, offset]).fetchall()
    conn.close()
    
    # Ensure page is valid
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    return render_template(
        'index.html',
        rows=rows,
        total=total,
        page=page,
        total_pages=total_pages,
        filter_val=filter_val,
        threshold=threshold
    )

@app.route('/image/<int:image_id>')
def image_detail(image_id):
    """Show detailed view of a single image with OCR results."""
    threshold = get_threshold()
    conn = get_db()
    
    # Fetch image
    img = conn.execute(
        "SELECT * FROM images WHERE id = ?",
        (image_id,)
    ).fetchone()
    
    if not img:
        conn.close()
        return "Image not found", 404
    
    # Fetch OCR results
    ocr = conn.execute(
        "SELECT * FROM ocr_results WHERE image_id = ? ORDER BY created_at DESC",
        (image_id,)
    ).fetchall()
    
    conn.close()
    
    return render_template(
        'image_detail.html',
        img=img,
        ocr=ocr,
        threshold=threshold
    )

@app.route('/photo/<int:image_id>')
def serve_photo(image_id):
    """Serve the actual photo file."""
    conn = get_db()
    img = conn.execute(
        "SELECT file_path FROM images WHERE id = ?",
        (image_id,)
    ).fetchone()
    conn.close()
    
    if not img:
        return "Not found", 404
    
    file_path = img['file_path']
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    return send_file(file_path, mimetype=mime_type or 'image/jpeg')

@app.route('/image/<int:image_id>/review', methods=['POST'])
def toggle_review(image_id):
    """Toggle the 'is_reviewed' status of an image (AJAX or form)."""
    conn = get_db()
    
    # Get current status
    img = conn.execute(
        "SELECT is_reviewed FROM images WHERE id = ?",
        (image_id,)
    ).fetchone()
    
    if not img:
        conn.close()
        return jsonify({"error": "Image not found"}), 404
    
    # Toggle
    new_status = 1 - img['is_reviewed']
    conn.execute(
        "UPDATE images SET is_reviewed = ? WHERE id = ?",
        (new_status, image_id)
    )
    conn.commit()
    conn.close()
    
    # Return JSON if AJAX, otherwise redirect
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"is_reviewed": new_status})
    else:
        return_to = request.form.get('return_to', 'index')
        return redirect(url_for('index') if return_to == 'index' else url_for('image_detail', image_id=image_id))

# ── Routes: Settings ───────────────────────────────────────────────────

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Display and update recipe detection threshold."""
    conn = get_db()
    
    if request.method == 'POST':
        threshold = request.form.get('recipe_threshold', '0.75')
        try:
            threshold = float(threshold)
            threshold = max(0.0, min(1.0, threshold))  # Clamp 0-1
        except ValueError:
            threshold = 0.75
        
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ('recipe_threshold', str(threshold))
        )
        conn.commit()
        conn.close()
        return redirect(url_for('settings'))
    
    threshold = get_threshold()
    conn.close()
    
    return render_template('settings.html', threshold=threshold)

# ── Routes: Admin & Pipeline ───────────────────────────────────────────

@app.route('/admin')
def admin():
    """Admin/pipeline control page."""
    global ocr_running
    conn = get_db()
    
    # Fetch stats
    stats = {}
    stats['images'] = conn.execute("SELECT COUNT(*) as c FROM images").fetchone()['c']
    stats['ocr'] = conn.execute("SELECT COUNT(*) as c FROM ocr_results").fetchone()['c']
    
    threshold = get_threshold()
    stats['detected'] = conn.execute(
        "SELECT COUNT(*) as c FROM ocr_results WHERE recipe_score >= ?",
        (threshold,)
    ).fetchone()['c']
    stats['reviewed'] = conn.execute(
        "SELECT COUNT(*) as c FROM images WHERE is_reviewed = 1"
    ).fetchone()['c']
    stats['unreviewed_detected'] = conn.execute(
        "SELECT COUNT(*) as c FROM images i "
        "JOIN ocr_results o ON i.id = o.image_id "
        "WHERE i.is_reviewed = 0 AND o.recipe_score >= ?",
        (threshold,)
    ).fetchone()['c']
    
    # Last OCR run
    last_run = conn.execute(
        "SELECT * FROM ocr_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    
    # Last OCR log lines
    last_log = []
    if last_run:
        log_rows = conn.execute(
            "SELECT line FROM ocr_log_lines WHERE run_id = ? ORDER BY ts ASC",
            (last_run['id'],)
        ).fetchall()
        last_log = [row['line'] for row in log_rows]
    
    conn.close()
    
    return render_template(
        'admin.html',
        stats=stats,
        last_run=last_run,
        last_log=last_log,
        ocr_running=ocr_running
    )

@app.route('/admin/run-ocr', methods=['POST'])
def run_ocr():
    """Start the OCR pipeline in a background thread."""
    global ocr_running, ocr_thread, ocr_lock
    
    with ocr_lock:
        if ocr_running:
            return "OCR already running", 409
        
        ocr_running = True
    
    def ocr_worker():
        global ocr_running
        try:
            conn = get_db()
            # Create OCR run record
            conn.execute("INSERT INTO ocr_runs (started_at, status) VALUES (CURRENT_TIMESTAMP, 'running')")
            conn.commit()
            run_id = conn.lastrowid
            conn.close()
            
            # Run OCR pipeline
            run_ocr_pipeline(run_id=run_id)
        finally:
            with ocr_lock:
                ocr_running = False
    
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()
    
    return "", 204

@app.route('/admin/ocr-stream')
def ocr_stream():
    """Server-Sent Events stream for OCR pipeline output."""
    def generate():
        conn = get_db()
        last_run = conn.execute(
            "SELECT id FROM ocr_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        
        if not last_run:
            yield "data: [error] No OCR run found\n\n"
            return
        
        run_id = last_run['id']
        last_line_id = 0
        
        # Stream new log lines as they arrive
        while True:
            conn = get_db()
            rows = conn.execute(
                "SELECT id, line FROM ocr_log_lines WHERE run_id = ? AND id > ? ORDER BY id ASC",
                (run_id, last_line_id)
            ).fetchall()
            conn.close()
            
            for row in rows:
                last_line_id = row['id']
                # Escape newlines for SSE
                line = row['line'].replace('\n', '\\n')
                yield f"data: {line}\n\n"
            
            # Check if pipeline is done
            conn = get_db()
            run_status = conn.execute(
                "SELECT status FROM ocr_runs WHERE id = ?",
                (run_id,)
            ).fetchone()
            conn.close()
            
            if run_status and run_status['status'] != 'running':
                yield "data: __DONE__\n\n"
                break
            
            # Short sleep to avoid busy-waiting
            import time
            time.sleep(0.1)
    
    return generate(), 200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }

@app.route('/admin/clear-scans', methods=['POST'])
def clear_scans():
    """Delete all images, OCR results, and run logs."""
    conn = get_db()
    conn.execute("DELETE FROM ocr_log_lines")
    conn.execute("DELETE FROM ocr_runs")
    conn.execute("DELETE FROM ocr_results")
    conn.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    return redirect(url_for('admin'))

@app.route('/admin/reset-db', methods=['POST'])
def reset_db():
    """Full database reset: delete all tables and reinitialize."""
    conn = get_db()
    
    # Drop all tables
    conn.execute("DROP TABLE IF EXISTS ocr_log_lines")
    conn.execute("DROP TABLE IF EXISTS ocr_runs")
    conn.execute("DROP TABLE IF EXISTS ocr_results")
    conn.execute("DROP TABLE IF EXISTS images")
    conn.execute("DROP TABLE IF EXISTS settings")
    
    conn.commit()
    conn.close()
    
    # Reinitialize database
    import init_db  # This will recreate all tables
    
    return redirect(url_for('admin'))

# ── Error Handlers ────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Server error"), 500

# ── Context Processors ───────────────────────────────────────────────

@app.context_processor
def inject_threshold():
    """Make threshold available to all templates."""
    return {'threshold': get_threshold()}

# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # Development server (use Gunicorn in production)
    app.run(host='0.0.0.0', port=5000, debug=False)