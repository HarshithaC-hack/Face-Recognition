# Eagle Access — Intelligent Facial Recognition System
Eagle Access is a modular, full-stack facial recognition access management system built from scratch — designed to evolve step-by-step from a Python core library to a production-ready web + desktop solution.

It combines computer vision, machine learning, and user-friendly interfaces to create a seamless experience for secure, automated entry management.

## Layout
```
eagle_app/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── user_manager.py
│   ├── photo_capture.py
│   ├── embedding_manager.py
│   ├── face_recognition.py
│   └── api.py                  # Flask routes (Phase 2)
├── frontend/
│   └── app_gui.py              # Tkinter GUI (Phase 3)
├── dataset/
│   └── Custom/                 # Captured user images
├── main_console.py             # CLI test runner
├── requirements.txt
└── README.md

```

## Quick start (CLI smoke test)
```bash
# 1) Create venv and install deps
pip install -r requirements.txt

# 2) Run the console menu to try Phase 1 without Flask/Tkinter
python main_console.py
# 3) Go to the frontend folder
cd frontend
python app_gui.py
