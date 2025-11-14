# snapchat.py
# requirements:
# pip install opencv-python pillow customtkinter numpy
# run from project folder with .venv activated

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path
import sys
import sqlite3
from datetime import datetime
import os
import json
import hashlib
import re

# ---------- CONFIG / TUNABLES ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
BASE = PROJECT_ROOT / "images"            # put overlays here
SAVED_DIR = PROJECT_ROOT / "saved"       # saved images go here
DB_PATH = PROJECT_ROOT / "snapchat.db"
CONFIG_PATH = PROJECT_ROOT / "config.json"

# placement tuning (tweak these if overlay sits off)
GLASSES_SCALE = 1.10
GLASSES_EYE_LEVEL = 0.16
MUSTACHE_SCALE = 0.55
MUSTACHE_Y = 0.62
NOSE_SCALE = 0.22
NOSE_Y = 0.42

FILES = {
    "glasses1": BASE / "glasses.png",
    "glasses2": BASE / "glasses1.png",
    "glasses3": BASE / "glasses2.png",
    "mustache": BASE / "cat_whiskers.png",
    "clown_nose": BASE / "clown_nose.png",
}
# party_hat removed as requested

# ---------- DB helpers ----------
SAVED_DIR.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS saved_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filter_name TEXT,
        username TEXT,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    # simple hashing for demo (not a production-ready KDF)
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def create_user(username: str, password: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?",
                (username, hash_password(password)))
    row = cur.fetchone()
    conn.close()
    return row is not None

def insert_saved_image_record(filename: str, filter_name: str, username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO saved_images (filename, filter_name, username, created_at) VALUES (?, ?, ?, ?)",
        (filename, filter_name, username, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_saved_images_for_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename, filter_name, created_at FROM saved_images WHERE username=? ORDER BY id DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_saved_image_record(image_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filename FROM saved_images WHERE id=?", (image_id,))
    row = cur.fetchone()
    if row:
        fname = row[0]
        # remove file if exists
        try:
            p = Path(fname)
            if p.exists():
                p.unlink()
        except Exception:
            pass
        cur.execute("DELETE FROM saved_images WHERE id=?", (image_id,))
        conn.commit()
    conn.close()

def get_filter_stats(username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filter_name, COUNT(*) FROM saved_images WHERE username=? GROUP BY filter_name", (username,))
    rows = cur.fetchall()
    conn.close()
    return {r[0] if r[0] else "None": r[1] for r in rows}

# init DB
init_db()

# ---------- config helpers (remember-me) ----------
def read_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}

def write_config(d: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(d))
    except Exception:
        pass

config = read_config()

# ---------- UTIL ----------
def load_rgba(path: Path):
    """Load an RGBA image with cv2 and warn if missing."""
    if not path.exists():
        print(f"Warning: missing file {path}", file=sys.stderr)
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: cv2 couldn't read {path}", file=sys.stderr)
    return img

overlays = {name: load_rgba(p) for name, p in FILES.items()}

print("Loaded overlays:")
for k, v in overlays.items():
    print(f"  {k}: {'OK' if v is not None else 'MISSING'}")

# face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def overlay_rgba(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Clip overlay region to background bounds
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(bw, x + ow); y2 = min(bh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return background

    ox1 = x1 - x; oy1 = y1 - y
    ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)

    region = overlay[oy1:oy2, ox1:ox2]
    bg_region = background[y1:y2, x1:x2].astype(float)

    # If overlay has no alpha channel, just copy RGB
    if region.shape[2] < 4:
        background[y1:y2, x1:x2] = region[:, :, :3]
        return background

    alpha = region[:, :, 3].astype(float) / 255.0
    alpha = np.expand_dims(alpha, 2)  # shape (h,w,1)
    fg = region[:, :, :3].astype(float)

    comp = fg * alpha + bg_region * (1 - alpha)
    background[y1:y2, x1:x2] = comp.astype(np.uint8)
    return background

# ---------- FILTER LOGIC ----------
def apply_filter(frame, selected_filter):
    if selected_filter is None:
        return frame
    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        overlay = overlays.get(selected_filter)
        if overlay is None:
            continue

        # TUNED placement & scaling per filter type
        if selected_filter.startswith("glasses"):
            gw = int(w * GLASSES_SCALE)
            gh = int(overlay.shape[0] * (gw / max(1, overlay.shape[1])))
            fx = x - int((gw - w) / 2)
            fy = y + int(h * GLASSES_EYE_LEVEL)

        elif selected_filter == "mustache":
            gw = int(w * MUSTACHE_SCALE)
            gh = int(overlay.shape[0] * (gw / max(1, overlay.shape[1])))
            fx = x + int((w - gw) / 2)
            fy = y + int(h * MUSTACHE_Y)

        elif selected_filter == "clown_nose":
            gw = int(w * NOSE_SCALE)
            gh = int(overlay.shape[0] * (gw / max(1, overlay.shape[1])))
            fx = x + int((w - gw) / 2)
            fy = y + int(h * NOSE_Y)

        else:
            gw = int(w)
            gh = int(overlay.shape[0] * (gw / max(1, overlay.shape[1])))
            fx, fy = x, y

        # Resize and overlay
        try:
            resized = cv2.resize(overlay, (max(1, gw), max(1, gh)), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = overlay

        out = overlay_rgba(out, resized, fx, fy)

    return out

# ---------- UI ----------
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Snapchat Filter App")
root.geometry("920x700+200+80")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
selected_filter = None
current_user = None
_update_running = False   # guard so update_frame starts only once

# Top status: logged in user label
top_frame = ctk.CTkFrame(root)
top_frame.pack(fill="x", pady=(6,0))
status_label = ctk.CTkLabel(top_frame, text="Not logged in", anchor="w")
status_label.pack(side="left", padx=8, pady=4)

label_video = ctk.CTkLabel(root, text="")
label_video.pack(pady=10, expand=True, fill="both")

frame_buttons = ctk.CTkFrame(root)
frame_buttons.pack(pady=10)

buttons = [
    ("Glasses 1", "glasses1"),
    ("Glasses 2", "glasses2"),
    ("Glasses 3", "glasses3"),
    ("Mustache", "mustache"),
    ("Clown Nose", "clown_nose"),
]

def set_filter(f):
    global selected_filter
    selected_filter = f

for i, (lbl, val) in enumerate(buttons):
    btn = ctk.CTkButton(frame_buttons, text=lbl, command=lambda v=val: set_filter(v))
    btn.grid(row=i//3, column=i%3, padx=10, pady=6)

# Save image (records username & filter)
def save_image():
    global current_user
    if current_user is None:
        show_message("Not logged in", "Please login to save images.")
        return
    ret, frame = cap.read()
    if not ret:
        print("No frame to save")
        return
    out = apply_filter(frame, selected_filter)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"saved_{current_user}_{ts}.jpg"
    fpath = SAVED_DIR / fname
    cv2.imwrite(str(fpath), out)
    insert_saved_image_record(str(fpath), selected_filter, current_user)
    print(f"Saved {fpath} with filter={selected_filter}")

btn_save = ctk.CTkButton(root, text="Save Image", fg_color="green", command=save_image)
btn_save.pack(pady=6)
# disabled until logged in
btn_save.configure(state="disabled")

# small helper message window (safer than CTkMessageBox which may not exist)
def show_message(title: str, message: str):
    w = ctk.CTkToplevel(root)
    w.title(title)
    w.geometry("320x140")
    lbl = ctk.CTkLabel(w, text=message, wraplength=280)
    lbl.pack(expand=True, padx=12, pady=12)
    b = ctk.CTkButton(w, text="OK", command=w.destroy)
    b.pack(pady=8)
    w.grab_set()

# Gallery
def show_gallery():
    global current_user
    if current_user is None:
        show_message("Not logged in", "Please login to view your gallery.")
        return
    gallery = ctk.CTkToplevel(root)
    gallery.title(f"{current_user} - Gallery")
    gallery.geometry("900x500")
    frame_list = ctk.CTkScrollableFrame(gallery)
    frame_list.pack(fill="both", expand=True, padx=8, pady=8)

    rows = get_saved_images_for_user(current_user)
    if not rows:
        lbl = ctk.CTkLabel(frame_list, text="No saved images yet.")
        lbl.pack(pady=8)
        return

    for r in rows:
        img_id, filename, filter_name, created_at = r
        rowf = ctk.CTkFrame(frame_list)
        rowf.pack(fill="x", pady=6, padx=6)

        # thumbnail
        try:
            pil = Image.open(filename)
            pil.thumbnail((160, 120))
            tkimg = ImageTk.PhotoImage(pil)
            lbl_img = ctk.CTkLabel(rowf, image=tkimg, text="")
            lbl_img.image = tkimg
            lbl_img.pack(side="left", padx=6)
        except Exception:
            lbl_img = ctk.CTkLabel(rowf, text="[Image error]")
            lbl_img.pack(side="left", padx=6)

        info = f"Filter: {filter_name}\nSaved: {created_at}\nID: {img_id}"
        lbl_info = ctk.CTkLabel(rowf, text=info, width=300)
        lbl_info.pack(side="left", padx=6)

        def make_delete(iid):
            def _d():
                delete_saved_image_record(iid)
                gallery.destroy()
                show_gallery()
            return _d

        btn_del = ctk.CTkButton(rowf, text="Delete", fg_color="red", command=make_delete(img_id))
        btn_del.pack(side="right", padx=6)

# Stats
def show_stats():
    global current_user
    if current_user is None:
        show_message("Not logged in", "Please login to view stats.")
        return
    stats = get_filter_stats(current_user)
    s = "\n".join(f"{k or 'None'}: {v}" for k, v in stats.items()) or "No data yet."
    stat_win = ctk.CTkToplevel(root)
    stat_win.title("Stats")
    stat_win.geometry("300x200")
    ctk.CTkLabel(stat_win, text=f"Stats for {current_user}", font=("Arial", 14)).pack(pady=8)
    ctk.CTkLabel(stat_win, text=s).pack(padx=10, pady=6)

# Logout
def do_logout():
    global current_user
    current_user = None
    status_label.configure(text="Not logged in")
    btn_save.configure(state="disabled")
    show_message("Logged out", "You have been logged out.")

# Buttons for gallery/stats/login/logout
extra_frame = ctk.CTkFrame(root)
extra_frame.pack(pady=6)
btn_gallery = ctk.CTkButton(extra_frame, text="Gallery", command=show_gallery)
btn_gallery.grid(row=0, column=0, padx=6)
btn_stats = ctk.CTkButton(extra_frame, text="Stats", command=show_stats)
btn_stats.grid(row=0, column=1, padx=6)
btn_logout = ctk.CTkButton(extra_frame, text="Logout", fg_color="orange", command=do_logout)
btn_logout.grid(row=0, column=2, padx=6)

# ---------- Auth modal (improved) ----------
def password_strength(pw: str) -> str:
    """Return 'Weak', 'Medium', 'Strong'."""
    if not pw:
        return ""
    score = 0
    if len(pw) >= 8: score += 1
    if re.search(r"[0-9]", pw): score += 1
    if re.search(r"[A-Z]", pw): score += 1
    if re.search(r"[a-z]", pw): score += 1
    if re.search(r"[^A-Za-z0-9]", pw): score += 1
    if score <= 2:
        return "Weak"
    if score <= 4:
        return "Medium"
    return "Strong"

def show_login_modal():
    global current_user, _update_running, config
    modal = ctk.CTkToplevel(root)
    modal.title("Login / Register")
    modal.geometry("380x300")
    modal.grab_set()
    modal.transient(root)

    lbl = ctk.CTkLabel(modal, text="Login or Register", font=("Arial", 16))
    lbl.pack(pady=8)

    user_var = ctk.StringVar()
    pass_var = ctk.StringVar()
    remember_var = ctk.BooleanVar(value=bool(config.get("remembered_username")))

    # prefill username if remembered
    if config.get("remembered_username"):
        user_var.set(config.get("remembered_username"))

    ent_user = ctk.CTkEntry(modal, placeholder_text="username", textvariable=user_var)
    ent_user.pack(pady=6, padx=12, fill="x")

    pw_frame = ctk.CTkFrame(modal)
    pw_frame.pack(fill="x", padx=12, pady=6)
    ent_pass = ctk.CTkEntry(pw_frame, placeholder_text="password", textvariable=pass_var, show="*")
    ent_pass.pack(side="left", expand=True, fill="x")

    # show/hide password
    def toggle_show():
        if ent_pass.cget("show") == "":
            ent_pass.configure(show="*")
            btn_show.configure(text="Show")
        else:
            ent_pass.configure(show="")
            btn_show.configure(text="Hide")

    btn_show = ctk.CTkButton(pw_frame, text="Show", width=60, command=toggle_show)
    btn_show.pack(side="right", padx=(6,0))

    # password strength indicator
    strength_lbl = ctk.CTkLabel(modal, text="", fg_color=None)
    strength_lbl.pack(pady=(0,6))

    def on_pw_change(*args):
        s = password_strength(pass_var.get())
        if s == "Weak":
            strength_lbl.configure(text="Password strength: Weak")
        elif s == "Medium":
            strength_lbl.configure(text="Password strength: Medium")
        elif s == "Strong":
            strength_lbl.configure(text="Password strength: Strong")
        else:
            strength_lbl.configure(text="")

    pass_var.trace_add("write", lambda *a: on_pw_change())

    remember_chk = ctk.CTkCheckBox(modal, text="Remember username", variable=remember_var)
    remember_chk.pack(pady=6)

    msg_lbl = ctk.CTkLabel(modal, text="")
    msg_lbl.pack(pady=6)

    def do_register():
        u = user_var.get().strip()
        p = pass_var.get()
        if not u or not p:
            msg_lbl.configure(text="Enter username and password")
            return
        if len(p) < 6:
            msg_lbl.configure(text="Password too short (min 6 chars)")
            return
        ok = create_user(u, p)
        if ok:
            msg_lbl.configure(text="Registered — you can login now")
        else:
            msg_lbl.configure(text="Username exists")

    def do_login():
        nonlocal_modal_actions = None  # placeholder to keep structure clear
        global current_user, _update_running, config
        u = user_var.get().strip()
        p = pass_var.get()
        if not u or not p:
            msg_lbl.configure(text="Enter username and password")
            return
        if verify_user(u, p):
            current_user = u
            # remember username if checked
            if remember_var.get():
                config["remembered_username"] = u
            else:
                config.pop("remembered_username", None)
            write_config(config)
            modal.grab_release()
            modal.destroy()
            status_label.configure(text=f"Logged in as: {current_user}")
            btn_save.configure(state="normal")
            print(f"User logged in: {current_user}")
            # ensure update loop is running (start only once)
            if not _update_running:
                _update_running = True
                root.after(0, update_frame)
        else:
            msg_lbl.configure(text="Login failed")

    btn_frame = ctk.CTkFrame(modal)
    btn_frame.pack(pady=6)
    btn_reg = ctk.CTkButton(btn_frame, text="Register", command=do_register)
    btn_reg.grid(row=0, column=0, padx=8)
    btn_login = ctk.CTkButton(btn_frame, text="Login", command=do_login)
    btn_login.grid(row=0, column=1, padx=8)

    ent_pass.bind("<Return>", lambda e: do_login())

# Show login modal at start
show_login_modal()

# ---------- video update ----------
def update_frame():
    global _update_running
    ret, frame = cap.read()
    if not ret:
        root.after(50, update_frame)
        return
    out = apply_filter(frame, selected_filter)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(out_rgb)
    imgtk = ImageTk.PhotoImage(img)
    label_video.img = imgtk
    label_video.configure(image=imgtk)
    root.after(15, update_frame)

def on_exit():
    try:
        cap.release()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_exit)
root.mainloop()
