# Thermal Vision Monitoring Platform


<img width="1101" height="612" alt="Screenshot 2026-02-18 205325" src="https://github.com/user-attachments/assets/e1b7ac00-7401-4218-baed-97948eb5abef" />


Thermal Vision is a multi-module Flask platform fusing thermal object detection, user lifecycle management, and AI incident narration. Ultralytics YOLOv9 continuously analyzes a thermal feed, structured events are written to `run_structured.jsonl`, and a Gemini-powered agent summarizes activity while falling back to raw logs when rate limits occur. Each app blueprint (`app1`-`app7`) hosts a focused experience ranging from dashboards (`app1`) to auth flows (`app2`) and auxiliary analysis tools.

### Core Capabilities
- Real-time person localization on thermal video via YOLOv9 (`app3`).
- Structured log persistence plus optional text-to-speech briefings through `pyttsx3`.
- AI summaries using Google Gemini when `GOOGLE_API_KEY` or `GEMINI_API_KEY` is present; automatic raw-log fallback otherwise.
- MySQL-backed auth and verification (database name `thermoai_user_db`) exposed through Flask blueprints.
- Modular route registration from `run.py`, giving `/appN` namespaces for each feature module.

### Technology Stack
- Python 3.10, Flask 3.1, Flask-SQLAlchemy 3.1
- MySQL + `mysql-connector-python` 9.6
- Ultralytics YOLOv9 (`yolov9c.pt`) for detections
- Google Gemini (`google-genai`) + `pyttsx3` for AI narration/TTS
- Bootstrap-ready Jinja templates per app directory

### Quick Start
1. **Create & activate a venv** (PowerShell example):
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (either export or create `.env` beside `Project/`):
   ```
   SECRET_KEY=dev-secret
   MYSQL_HOST=localhost
   MYSQL_USER=root
   MYSQL_PASSWORD=yourpass
   MYSQL_DATABASE=thermoai_user_db
   GOOGLE_API_KEY=your-gemini-key
   SMTP_HOST=smtp.gmail.com
   SMTP_USER=...
   SMTP_PASSWORD=...
   ```
4. **Provision the database** `thermoai_user_db` and run any migrations/SQL for your user tables.
5. **Run the server** from the workspace root:
   ```
   python "Project\run.py"
   ```
6. **Navigate**: open `http://127.0.0.1:5000/app2/login` to authenticate, then explore `/app1` through `/app7`. The console will emit `AGENT | ...` lines when AI summaries succeed, or `[RAW ACTIVITY LOG]` when the agent falls back because of rate limits or missing keys.

### Thermal Dataset Notes
- Recommended source: [FLIR ADAS dataset](https://oem.flir.com/en-IN/solutions/automotive/adas-dataset-form/), offering 16-bit thermal + RGB images with human bounding boxes.
- Preprocess by normalizing 16-bit images to 0–255, convert to 8-bit, then resize to 640×640 for YOLO.
- Baseline training recipe: image size 640, batch 8–16, 50–100 epochs, default YOLO optimizer, 80/20 train/val split.

Extend each blueprint with new dashboards or analytics modules as needed—`run.py` automatically wires them through their `create_app()` factories.
