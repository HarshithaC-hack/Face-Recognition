"""
Phase 3 — Tkinter frontend for Eagle Access
-------------------------------------------
Stylish desktop GUI that communicates with the Flask backend API.
Now includes clean log formatting, timestamps, sequential actions,
and live registration completion tracking.
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
import requests
import json
import threading
import time
from datetime import datetime

BACKEND_URL = "http://127.0.0.1:5000"


class EagleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🦅 Eagle Access System")
        self.geometry("560x540")
        self.config(bg="#1c1c1c")

        # --- Heading ---
        tk.Label(
            self,
            text="🦅 Eagle Access Control",
            font=("Segoe UI", 20, "bold"),
            fg="#ffffff",
            bg="#1c1c1c",
        ).pack(pady=15)

        # --- Name Entry ---
        frame = tk.Frame(self, bg="#1c1c1c")
        frame.pack(pady=10)
        tk.Label(
            frame,
            text="User Name:",
            font=("Segoe UI", 11),
            fg="#ffffff",
            bg="#1c1c1c",
        ).pack(side=tk.LEFT)
        self.name_entry = tk.Entry(frame, width=30, font=("Segoe UI", 11))
        self.name_entry.pack(side=tk.LEFT, padx=10)

        # --- Buttons ---
        btn_frame = tk.Frame(self, bg="#1c1c1c")
        btn_frame.pack(pady=10)

        self._make_btn(btn_frame, "Register User", self.register_user, "#38b000").pack(side=tk.LEFT, padx=6)
        self._make_btn(btn_frame, "Delete User", self.delete_user, "#e63946").pack(side=tk.LEFT, padx=6)
        self._make_btn(btn_frame, "List Users", self.list_users, "#4361ee").pack(side=tk.LEFT, padx=6)
        self._make_btn(btn_frame, "Access Eagle", self.access_eagle, "#fcbf49").pack(side=tk.LEFT, padx=6)

        # --- Output Log ---
        tk.Label(
            self,
            text="System Logs",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#1c1c1c",
        ).pack(pady=(10, 2))
        self.output = scrolledtext.ScrolledText(
            self,
            width=65,
            height=17,
            bg="#202020",
            fg="#00ff00",
            font=("Consolas", 10),
            insertbackground="#00ff00",
        )
        self.output.pack(padx=10, pady=5)

        # --- Footer ---
        tk.Label(
            self,
            text="Phase 3 – Powered by Flask + Tkinter",
            font=("Segoe UI", 9),
            fg="#888888",
            bg="#1c1c1c",
        ).pack(side=tk.BOTTOM, pady=6)

    # ---------------------------------------------------------
    # Button Factory
    # ---------------------------------------------------------
    def _make_btn(self, parent, text, command, color):
        return tk.Button(
            parent,
            text=text,
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            width=12,
            cursor="hand2",
            command=command,
        )

    # ---------------------------------------------------------
    # Backend Interaction (API Calls)
    # ---------------------------------------------------------
    def register_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing Info", "Please enter a user name first.")
            return
        self._log("⏳ Starting registration...")
        try:
            res = requests.post(f"{BACKEND_URL}/register", data={"name": name}, timeout=60)
            self._log(res.text)
            # Start polling for registration completion
            threading.Thread(target=self._poll_registration_status, args=(name,), daemon=True).start()
        except requests.exceptions.ConnectionError:
            self._log("[Error] Backend not reachable. Is Flask running?")
        except Exception as e:
            self._log(f"[Error] {e}")

    def _poll_registration_status(self, name):
        """Continuously poll /status/<name> every 3s until complete."""
        while True:
            try:
                res = requests.get(f"{BACKEND_URL}/status/{name}")
                data = res.json()
                status = data.get("status")
                if status == "completed":
                    self._log(f"✅ Registration complete for {name}")
                    break
                elif status == "failed":
                    self._log(f"❌ Registration failed for {name}")
                    break
            except Exception:
                pass
            time.sleep(3)

    def delete_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing Info", "Please enter a user name.")
            return
        self._log("🗑️ Deleting user...")
        try:
            res = requests.delete(f"{BACKEND_URL}/delete/{name}", timeout=15)
            self._log(res.text)
            if res.status_code == 200:
                self._log("🔁 Refreshing user list...")
                res = requests.get(f"{BACKEND_URL}/list")
                self._log(res.text)
        except requests.exceptions.ConnectionError:
            self._log("[Error] Backend not reachable. Is Flask running?")
        except Exception as e:
            self._log(f"[Error] {e}")

    def list_users(self):
        self._log("📋 Fetching user list...")
        try:
            res = requests.get(f"{BACKEND_URL}/list", timeout=15)
            self._log(res.text)
        except requests.exceptions.ConnectionError:
            self._log("[Error] Backend not reachable. Is Flask running?")
        except Exception as e:
            self._log(f"[Error] {e}")

    def access_eagle(self):
        self._log("🧠 Running face verification...")
        try:
            res = requests.post(f"{BACKEND_URL}/access", timeout=90)
            self._log(res.text)
        except requests.exceptions.ConnectionError:
            self._log("[Error] Backend not reachable. Is Flask running?")
        except Exception as e:
            self._log(f"[Error] {e}")

    # ---------------------------------------------------------
    # Log Formatting + Display
    # ---------------------------------------------------------
    def _format_response(self, text: str) -> str:
        """Beautify JSON or plain text API responses."""
        try:
            data = json.loads(text)
        except Exception:
            return text
        if isinstance(data, dict):
            if "status" in data and "confidence" in data:
                conf = float(data.get("confidence", 0)) * 100
                if data["status"].lower() == "granted":
                    return (
                        f"✅ ACCESS GRANTED to {data.get('name', '')}\n"
                        f"Confidence: {conf:.2f}%\n"
                        f"Time: {data.get('time', '')}"
                    )
                else:
                    return (
                        f"❌ ACCESS DENIED\n"
                        f"Confidence: {conf:.2f}%\n"
                        f"Time: {data.get('time', '')}"
                    )
            if "message" in data:
                return f"🟢 {data['message']}"
            if "error" in data:
                return f"🔴 Error: {data['error']}"
            if all(isinstance(v, dict) and "name" in v for v in data.values()):
                users = "\n".join([f"• {v['name']} (ID: {k})" for k, v in data.items()])
                return f"👥 Registered Users\n{users}"
        return json.dumps(data, indent=2)

    def _log(self, text: str):
        """Insert formatted log with timestamp."""
        now = datetime.now().strftime("%H:%M:%S")
        formatted = self._format_response(text)
        self.output.insert(tk.END, f"[{now}] {formatted}\n" + ("-" * 55) + "\n")
        self.output.see(tk.END)


# -------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------
if __name__ == "__main__":
    app = EagleApp()
    app.mainloop()
