# PHONEIX V9 is Thermal Vision Monitoring Platform   


<img width="1906" height="910" alt="image" src="https://github.com/user-attachments/assets/c845b2ff-620e-4dcb-acf5-87023028585b" />


Welcome to NightFlare!

NightFlare is a next-generation surveillance system designed to keep you safe and informed, day and night. Our platform combines advanced technology with easy-to-understand features, making it accessible for everyone.




**Technology Stack**

- Python 3.10
- Flask 3.1
- Flask-SQLAlchemy 3.1
- MySQL + mysql-connector-python 9.6
- Ultralytics YOLOv9c (thermal detection)
- Ultralytics YOLOv8 (RGB detection)
- Google Gemini (AI summaries)
- pyttsx3 (text-to-speech)
- Bootstrap-ready Jinja templates



**What problem does Thermal Vision solve?**

Traditional surveillance systems often struggle in challenging conditions—darkness, smoke, fog, or when rapid response is needed for emergencies like wildfires. Thermal Vision overcomes these limitations by using smart thermal imaging and AI-driven detection, ensuring you never miss a critical event. Whether it’s monitoring remote areas, detecting people in low visibility, or spotting wildfires before they spread, our system provides reliable protection where others fall short.


<img width="1880" height="904" alt="image" src="https://github.com/user-attachments/assets/e86a3dab-2c44-47db-b0a5-b2aa58cbe324" />

**Where and why is it useful?**

Thermal Vision is ideal for:
- Forests and wildland areas, for early wildfire detection
- Industrial sites, to monitor safety and prevent accidents
- Residential and commercial properties, for 24/7 security
- Public spaces, to track movement and ensure safety
- Any location where visibility is poor or risks are high

By combining agentic monitoring, multi-model detection for day and night, and specialized wildfire surveillance, Thermal Vision delivers peace of mind and actionable alerts—no matter the environment or situation.
### Features

nference with YOLOv9 on thermal footage (gray + thermal colormap views)

2.Real-time detection overlays with labels, confidence, and styling

3.Per-frame detection logs with size (width/height) and confidence

4.Person tracking across frames with stable ID

5.Motion state classification for people (standing / moving)
Human count logging (per frame + total)

6. Density-over-time (rolling average of people in scene)

7.Anomaly alerts:
Thermal spike on a person (possible fire proximity)
**What makes Thermal Vision special?**

- **Agentic Monitoring:** Our system acts like a smart agent, constantly watching over your environment and alerting you to any unusual activity. It’s always on guard, so you don’t have to be.

- **Multi-Model Detection:** We use a combination of models to ensure reliable detection in all conditions:
   - **Day and Night Detection:** Whether it’s bright daylight or pitch-black night, our system adapts to provide clear and accurate monitoring.
   - **Special Thermal Imaging for Wildfire Detection:** Our unique thermal imaging technology is specially designed to spot wildfires early, helping prevent disasters before they spread.

- **Super Surveillance:** By combining these features, Thermal Vision offers a powerful, all-in-one solution for security and safety. It’s like having a team of experts watching over your property, ready to respond to any threat.

**How it works:**

Thermal Vision uses smart cameras and sensors to monitor your surroundings. When something unusual is detected—like a person, fire, or movement—the system sends you an alert. You can check live feeds, review past events, and stay informed from anywhere.


<img width="1101" height="612" alt="Screenshot 2026-02-18 205325" src="https://github.com/user-attachments/assets/e1b7ac00-7401-4218-baed-97948eb5abef" />


**Why choose Thermal Vision?**

- Easy to use, no technical expertise required
- Works in all weather and lighting conditions—even in darkness, smoke, or fog
- Early warning for wildfires and other emergencies
- Peace of mind, knowing you’re always protected


If you want a surveillance system that’s smart, reliable, and ready for anything, Thermal Vision is the answer.

For more information or help, please contact our support team.

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
