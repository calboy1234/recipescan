RecipeScan Documentation

Project Overview
RecipeScan is an automated OCR pipeline designed for high-throughput processing of hundreds of thousands of images. It extracts, scores, and stores recipe text, optimizing for "recipe-ness" detection. A Flask-based web interface is provided for monitoring, managing, and interacting with the pipeline and its results.

Core Components
*   `ocr_pipeline.py`: The production-grade engine. It features:
    *   Bounded Memory: Processes images in chunks to prevent memory bloat.
    *   Parallelism: Managed by a ThreadPoolExecutor with a strict OMP_NUM_THREADS=1 constraint to avoid thread contention.
    *   Pre-processing: Uses a global grayscale threshold (gray_thresh160) for optimal contrast on ink-on-paper images.
    *   OSD: Automated rotation detection using grayscale imagery before binarization.
    *   Metadata Extraction: Uses ExifTool to capture "Date Taken" (captured_at), falling back to file modification time if EXIF is missing.
*   `Tesseract_Testing_Suite/`: A nested grid-search repository used to find optimal Tesseract parameters (PSM, OEM, Scaling, etc.). It calculates a composite "Z-score" to identify configurations that perform consistently well across varied image sets.
*   **Web Interface (Flask App):**
    *   Built using Flask, it serves as the primary user interface.
    *   Provides endpoints for:
        *   Gallery view of scanned images with "Session Grouping" (clustering images taken within a 5-minute window).
        *   Detailed view of individual images, including per-run OCR parameters (PSM, OEM) and Run IDs.
        *   Statistics Dashboard: Interactive Chart.js visualizations for scan coverage, recipe score distribution, and pipeline history.
        *   Admin dashboard for monitoring pipeline status, history, and statistics.
        *   Starting, stopping, and managing OCR runs (including retrying failed images).
        *   System configuration and database integrity checks.
        *   Exporting processed data.
    *   Utilizes `templates/` and `static/` directories for its UI.
    *   Integrates with `ocr_pipeline.py` and `recipe_detector.py`.

Repository & Git Configuration
The project uses a nested Git structure to separate the production logic from the experimentation suite:

1.  **Root Repository (/recipescan)**
    *   Purpose: Manages the production pipeline (`ocr_pipeline.py`), the web interface (`app.py`, `templates/`, `static/`), database handling (`init_db.py`), and scoring logic (`recipe_detector.py`).
    *   `.gitignore`: Excludes artifacts, cached database files, and the testing suite folder to prevent experimental noise from entering production history.

2.  **Nested Repository (/Tesseract_Testing_Suite)**
    *   Purpose: A self-contained laboratory for OCR parameter optimization.
    *   Independence: It tracks its own history and configuration settings (e.g., `tesseract_testing_suite.py`) independently of the root repository.
    *   `.gitignore`: Tailored for experiment data (TSV logs, run folders, and cached image results).

Alignment & Maintenance
The system ensures consistency between experimentation and production by:
1.  Text Reconstruction: The testing suite mimics the production pipeline's image_to_data reconstruction logic to ensure scoring is based on the exact layout stored in the production DB.
2.  Web Interface Integration: The web interface allows for the monitoring and management of the OCR pipeline, providing visibility into its performance and results.
