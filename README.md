# Eagle Access — Phase 1 (Refactor to Modules)

This is a refactor of your facial-recognition access project into **importable Python modules**.
Designed for **Phase 1** of your full‑stack plan: make the core logic reusable so Flask (backend) and Tkinter (frontend) can call it cleanly in later phases.

## What changed?
- **No more subprocess calls** between scripts.
- Each concern is in its **own module** under `backend/`.
- Paths, thresholds, and knobs live in **`config.py`** (overridable via env vars).
- Functions are **documented** and designed to be imported from Flask or a GUI.

## Layout
```
eagle_app_phase1/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── user_manager.py
│   ├── photo_capture.py
│   ├── embedding_manager.py
│   └── face_recognition.py
├── dataset/Custom/                # captured data will be stored here
├── main_console.py                # optional CLI to smoke‑test modules
├── requirements.txt
└── README.md
```

## Quick start (CLI smoke test)
```bash
# 1) Create venv and install deps
pip install -r requirements.txt

# 2) Run the console menu to try Phase 1 without Flask/Tkinter
python main_console.py
```

> In **Phase 2**, you'll add `api.py` (Flask routes) and in **Phase 3** a Tkinter GUI calling those routes.
> These modules are already written to be imported directly by Flask/Tkinter later.
