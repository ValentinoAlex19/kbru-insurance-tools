import os
import re
import time
import uuid
import queue
import random
import hashlib
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, g
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
BINDER_FOLDER = os.path.join(BASE_DIR, 'binders')
DB_PATH = os.path.join(BASE_DIR, 'history.db')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimal 16MB total per request

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BINDER_FOLDER, exist_ok=True)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash",
    ).split(",")
    if m.strip()
]

MAX_RETRIES_PER_MODEL = int(os.getenv("MAX_RETRIES_PER_MODEL", "2"))
BASE_DELAY = 2
QUOTA_WAIT_SECONDS = int(os.getenv("QUOTA_WAIT_SECONDS", "30"))
RETRYABLE_CODES = {429, 500, 503, 504}
SKIP_MODEL_CODES = {404, 429}

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "20"))

PROMPT_VERSION = "v5"
MIN_SLOT_REQUIRED = (1, 2)
MAX_DOCS = 3
MAX_TEXT_CHARS = 300_000

PDF_IMAGE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}
TEXT_EXTENSIONS = {'.txt', '.docx'}
ALLOWED_EXTENSIONS = PDF_IMAGE_EXTENSIONS | TEXT_EXTENSIONS

TOOL_TYPES = {'tool1', 'tool2', 'tool3'}

# ---------------------------------------------------------------------------
# Prompt & Aturan
# ---------------------------------------------------------------------------
COMMON_RULES = (
    "\n\nATURAN WAJIB:\n"
    "- Setiap dokumen diberi label DOKUMEN 1, DOKUMEN 2, (dan DOKUMEN 3 bila ada) sebelum isinya. "
    "Gunakan label itu saat merujuk dokumen.\n"
    "- Gunakan hanya informasi yang benar-benar tertulis di dokumen. Jangan berasumsi atau mengarang.\n"
    "- Untuk setiap perbedaan/penyimpangan, sebutkan lokasinya (nomor pasal/klausul/halaman) "
    "dan kutip singkat isi aslinya dari masing-masing dokumen.\n"
    "- Jika suatu informasi tidak ditemukan atau tidak terbaca jelas, tulis 'Tidak ditemukan di dokumen'.\n"
    "- Tulis dalam Bahasa Indonesia yang rapi dan terstruktur (boleh memakai format Markdown dan tabel).\n"
    "- Kesimpulan adalah bantuan analisis dan perlu ditinjau oleh staf yang kompeten."
)

PROMPTS = {
    'tool1': (
        "Anda adalah Senior Insurance Underwriter dan Konsultan Ahli. "
        "Analisis dokumen Quotation asuransi ini (Perbandingan dokumen sejenis: Quotation Slip Lama vs Quotation Slip Renewal). "
        "Berikan output dalam format:\n"
        "1. Status Kesamaan TC (Apakah 100% sama atau ada perubahan?)\n"
        "2. Daftar Perbedaan Detail (Pasal/Klausul yang berubah, limit, deductible, dll)\n"
        "3. Kesimpulan Analisis: Apakah perubahan ini menguntungkan atau merugikan klien "
        "dibandingkan dokumen pertama?"
    ),
    'tool2': (
        "Anda adalah konsultan asuransi profesional. Analisis perbandingan dokumen tidak sejenis "
        "(Quotation Slip dengan Cover Note atau dengan Polis/Policy). Periksa apakah isi polis/cover note sudah sesuai komitmen di Quotation. "
        "Tampilkan poin kesamaan, perbedaan, serta kesimpulan untung/rugi bagi klien."
    ),
    'tool3': (
        "Anda adalah validator polis KBRU. Dokumen pertama adalah Master Binder KBRU, "
        "dan dokumen kedua adalah Quotation asuransi. "
        "Analisis apakah Quotation tersebut sudah sesuai paten dengan Master Binder KBRU. "
        "Tunjukkan penyimpangan dan simpulkan dampaknya bagi klien."
    ),
}

EXTRA_THIRD_DOC = {
    'tool1': (
        "Terdapat juga DOKUMEN 3 sebagai dokumen tambahan/pembanding. Bandingkan DOKUMEN 3 terhadap "
        "DOKUMEN 1 dan DOKUMEN 2, lalu tambahkan tabel perbandingan ketiga dokumen secara berdampingan."
    ),
    'tool2': (
        "Terdapat juga DOKUMEN 3 sebagai dokumen pendukung (misalnya endorsement atau dokumen tambahan). "
        "Periksa apakah DOKUMEN 3 konsisten dengan DOKUMEN 1 dan DOKUMEN 2, dan tunjukkan perbedaannya."
    ),
    'tool3': (
        "Terdapat juga DOKUMEN 3 berupa Quotation asuransi lain. Periksa kesesuaiannya dengan Master Binder KBRU "
        "(DOKUMEN 1), lalu bandingkan DOKUMEN 2 dan DOKUMEN 3 secara berdampingan dalam tabel."
    ),
}

def build_prompt(tool_type, n_docs):
    prompt = PROMPTS.get(tool_type, "Analisis dan bandingkan dokumen asuransi berikut secara komprehensif.")
    if n_docs >= 3:
        prompt += " " + EXTRA_THIRD_DOC.get(tool_type, "Bandingkan ketiga dokumen secara menyeluruh.")
    return prompt + COMMON_RULES

GEN_CONFIG = types.GenerateContentConfig(temperature=0.2)

# ---------------------------------------------------------------------------
# Database SQLite (Riwayat & Master Binder LOB)
# ---------------------------------------------------------------------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                tool_type TEXT,
                title TEXT,
                content_hash TEXT,
                result TEXT,
                model TEXT,
                created_at TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS binders (
                lob_name TEXT PRIMARY KEY,
                filename TEXT,
                file_path TEXT,
                file_kind TEXT,
                text_content TEXT,
                created_at TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_hist_client ON history(client_id, created_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_hist_hash ON history(client_id, content_hash)")

init_db()

# ---------------------------------------------------------------------------
# Cookies Client ID
# ---------------------------------------------------------------------------
COOKIE_NAME = "kbru_client"

@app.before_request
def identify_client():
    cid = request.cookies.get(COOKIE_NAME, "")
    g.set_cookie = False
    if not re.fullmatch(r"[0-9a-f]{32}", cid):
        cid = uuid.uuid4().hex
        g.set_cookie = True
    g.client_id = cid

@app.after_request
def persist_client_cookie(resp):
    if getattr(g, "set_cookie", False):
        resp.set_cookie(COOKIE_NAME, g.client_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax")
    return resp

# ---------------------------------------------------------------------------
# Helper File & Teks
# ---------------------------------------------------------------------------
def docx_to_text(path):
    try:
        import docx
        d = docx.Document(path)
        lines = []
        for p in d.paragraphs:
            if p.text.strip():
                lines.append(p.text.strip())
        return '\n'.join(lines)
    except Exception:
        return ""

def read_text_file(path, ext):
    if ext == '.txt':
        with open(path, 'rb') as fh:
            return fh.read().decode('utf-8', errors='replace')
    if ext == '.docx':
        return docx_to_text(path)
    return ""

def save_uploaded_file(file_storage, folder):
    original = secure_filename(file_storage.filename or "") or "dokumen"
    ext = os.path.splitext(original)[1].lower()
    if ext == '.doc':
        raise ValueError(f"{original}: format .doc lama tidak didukung. Simpan sebagai .docx atau PDF.")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"{original}: format tidak didukung.")
    path = os.path.join(folder, f"{uuid.uuid4().hex}{ext}")
    file_storage.save(path)
    return original, ext, path

# ---------------------------------------------------------------------------
# API LOB Master Binders (Tool 3)
# ---------------------------------------------------------------------------
@app.route('/api/lobs', methods=['GET'])
def list_lobs():
    with db() as c:
        rows = c.execute("SELECT lob_name, filename, created_at FROM binders ORDER BY lob_name ASC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/lobs', methods=['POST'])
def add_lob():
    lob_name = (request.form.get('lob_name') or "").strip()
    if not lob_name:
        return jsonify({"error": "Nama LOB wajib diisi"}), 400
    
    fs = request.files.get('binder_file')
    if not fs or not fs.filename:
        return jsonify({"error": "File Master Binder wajib diunggah"}), 400
    
    try:
        orig, ext, path = save_uploaded_file(fs, BINDER_FOLDER)
        text_content = ""
        kind = "file"
        if ext in TEXT_EXTENSIONS:
            text_content = read_text_file(path, ext).strip()
            os.remove(path)
            kind = "text"
            path = None

        with db() as c:
            old = c.execute("SELECT file_path FROM binders WHERE lob_name=?", (lob_name,)).fetchone()
            if old and old['file_path'] and os.path.exists(old['file_path']):
                try:
                    os.remove(old['file_path'])
                except Exception:
                    pass

            c.execute(
                "INSERT OR REPLACE INTO binders (lob_name, filename, file_path, file_kind, text_content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (lob_name, orig, path, kind, text_content, datetime.now(timezone.utc).isoformat(timespec='seconds'))
            )
        return jsonify({"ok": True, "lob_name": lob_name})
    except Exception as e:
        log.exception("Gagal menambah LOB")
        return jsonify({"error": str(e)}), 500

@app.route('/api/lobs/<lob_name>', methods=['DELETE'])
def delete_lob(lob_name):
    with db() as c:
        row = c.execute("SELECT file_path FROM binders WHERE lob_name=?", (lob_name,)).fetchone()
        if row and row['file_path'] and os.path.exists(row['file_path']):
            try:
                os.remove(row['file_path'])
            except Exception:
                pass
        cur = c.execute("DELETE FROM binders WHERE lob_name=?", (lob_name,))
    if not cur.rowcount:
        return jsonify({"error": "LOB tidak ditemukan"}), 404
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Analisis & Antrean
# ---------------------------------------------------------------------------
def prepare_slots_from_request(tool_type, saved_paths):
    slots = []
    time.sleep(0.5) # Jeda aman sistem tulis file
    
    if tool_type == 'tool3':
        lob_name = request.form.get('lob_name')
        if not lob_name:
            raise ValueError("Pilih LOB Master Binder terlebih dahulu.")
        with db() as c:
            binder = c.execute("SELECT * FROM binders WHERE lob_name=?", (lob_name,)).fetchone()
        if not binder:
            raise ValueError(f"Master Binder untuk LOB '{lob_name}' tidak ditemukan.")
        
        # Slot 1 adalah Binder
        if binder['file_kind'] == 'text':
            slots.append({"index": 1, "label": f"DOKUMEN 1 (Binder {lob_name})", "name": f"Master Binder {lob_name}", "kind": "text", "text": binder['text_content']})
        else:
            if binder['file_path'] and os.path.exists(binder['file_path']):
                saved_paths.append(binder['file_path'])
                slots.append({"index": 1, "label": f"DOKUMEN 1 (Binder {lob_name})", "name": binder['filename'], "kind": "file", "path": binder['file_path']})
            else:
                raise ValueError(f"File fisik Master Binder untuk LOB '{lob_name}' tidak ditemukan di server.")
        
        # Dokumen 2 & 3 untuk Tool 3
        for i in [2, 3]:
            fs = request.files.get(f"file{i}")
            text = (request.form.get(f"text{i}") or "").strip()
            if fs and fs.filename:
                orig, ext, path = save_uploaded_file(fs, UPLOAD_FOLDER)
                saved_paths.append(path)
                if ext in TEXT_EXTENSIONS:
                    txt = read_text_file(path, ext).strip()
                    os.remove(path)
                    slots.append({"index": i, "label": f"DOKUMEN {i}", "name": orig, "kind": "text", "text": txt})
                else:
                    slots.append({"index": i, "label": f"DOKUMEN {i}", "name": orig, "kind": "file", "path": path})
            elif text:
                slots.append({"index": i, "label": f"DOKUMEN {i}", "name": f"Teks Dokumen {i}", "kind": "text", "text": text})
    else:
        # Tool 1 & Tool 2 (Standar Dokumen 1, 2, 3)
        for i in range(1, MAX_DOCS + 1):
            fs = request.files.get(f"file{i}")
            text = (request.form.get(f"text{i}") or "").strip()
            if fs and fs.filename:
                orig, ext, path = save_uploaded_file(fs, UPLOAD_FOLDER)
                saved_paths.append(path)
                if ext in TEXT_EXTENSIONS:
                    txt = read_text_file(path, ext).strip()
                    os.remove(path)
                    slots.append({"index": i, "label": f"DOKUMEN {i}", "name": orig, "kind": "text", "text": txt})
                else:
                    slots.append({"index": i, "label": f"DOKUMEN {i}", "name": orig, "kind": "file", "path": path})
            elif text:
                slots.append({"index": i, "label": f"DOKUMEN {i}", "name": f"Teks Dokumen {i}", "kind": "text", "text": text})
    
    return slots

def compute_hash(tool_type, slots):
    h = hashlib.sha256()
    h.update(f"{PROMPT_VERSION}|{tool_type}|".encode())
    for s in slots:
        h.update(f"|{s['index']}:{s['kind']}:".encode())
        if s['kind'] == 'file':
            with open(s['path'], 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
        else:
            h.update(s['text'].encode('utf-8'))
    return h.hexdigest()

def make_title(slots):
    title = " vs ".join(s['name'] for s in slots)
    return title if len(title) <= 140 else title[:137] + "..."

def cleanup(local_paths, remote_files):
    for p in local_paths:
        try:
            if p and os.path.exists(p) and BINDER_FOLDER not in p:
                os.remove(p)
        except Exception:
            pass
    for f in remote_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

def error_code(err):
    code = getattr(err, "code", None)
    if isinstance(code, int):
        return code
    text = str(err)
    m = re.search(r'\b(404|429|500|503|504)\b', text)
    if m:
        return int(m.group(1))
    up = text.upper()
    if "RESOURCE_EXHAUSTED" in up:
        return 429
    if "UNAVAILABLE" in up or "OVERLOADED" in up:
        return 503
    return None

def wait_until_active(f, timeout=90):
    start = time.time()
    while True:
        state = getattr(f, "state", None)
        name = getattr(state, "name", str(state)) if state else ""
        if name == "FAILED":
            raise ValueError("Dokumen gagal diproses oleh AI.")
        if name != "PROCESSING":
            return f
        if time.time() - start > timeout:
            raise RuntimeError("Dokumen terlalu lama diproses oleh AI.")
        time.sleep(2)
        f = client.files.get(name=f.name)

def generate_with_fallback(contents):
    last_err = None
    for round_no in range(2):
        all_quota = True
        for model in MODELS:
            for attempt in range(MAX_RETRIES_PER_MODEL):
                try:
                    response = client.models.generate_content(
                        model=model, contents=contents, config=GEN_CONFIG
                    )
                    if not (response.text or "").strip():
                        raise RuntimeError("Respons AI kosong")
                    return response, model
                except Exception as err:
                    last_err = err
                    code = error_code(err)
                    if code in SKIP_MODEL_CODES:
                        if code != 429:
                            all_quota = False
                        break
                    all_quota = False
                    retryable = code in RETRYABLE_CODES or isinstance(err, RuntimeError)
                    if not retryable:
                        raise
                    if attempt < MAX_RETRIES_PER_MODEL - 1:
                        time.sleep(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
        if round_no == 0 and all_quota and QUOTA_WAIT_SECONDS > 0:
            time.sleep(QUOTA_WAIT_SECONDS)
            continue
        break
    raise last_err

def friendly_error(err):
    if isinstance(err, ValueError):
        return str(err)
    code = error_code(err)
    if code == 429:
        return "Kuota pemakaian AI sedang terlampaui. Coba lagi beberapa saat."
    if code in RETRYABLE_CODES or isinstance(err, RuntimeError):
        return "Layanan AI sedang sibuk. Sistem sudah mencoba model cadangan, silakan coba lagi."
    return "Terjadi kesalahan saat menganalisis dokumen."

job_queue = queue.Queue()
jobs = {}
waiting = []
jobs_lock = threading.Lock()

def enqueue_job(client_id, tool_type, slots, content_hash):
    job_id = uuid.uuid4().hex
    with jobs_lock:
        if len(waiting) >= MAX_QUEUE_SIZE:
            raise queue.Full()
        jobs[job_id] = {
            "client_id": client_id, "tool_type": tool_type, "slots": slots,
            "hash": content_hash, "status": "queued", "created_at": time.time(),
        }
        waiting.append(job_id)
        position = len(waiting)
    job_queue.put(job_id)
    return job_id, position

def run_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job: return
        if job_id in waiting: waiting.remove(job_id)
        job["status"] = "running"
        slots = job["slots"]

    remote_files = []
    local_paths = [s["path"] for s in slots if s["kind"] == "file"]

    try:
        contents = []
        for s in slots:
            if s["kind"] == "file":
                rf = client.files.upload(file=s["path"])
                rf = wait_until_active(rf)
                remote_files.append(rf)
                contents += [f"{s['label']} (file: {s['name']}):", rf]
            else:
                contents.append(f"{s['label']} ({s['name']}):\n{s['text']}\n--- akhir {s['label']} ---")

        contents.append(build_prompt(job["tool_type"], len(slots)))
        response, used_model = generate_with_fallback(contents)

        with db() as c:
            item_id = uuid.uuid4().hex
            created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            title = make_title(slots)
            c.execute("DELETE FROM history WHERE client_id=? AND content_hash=?", (job["client_id"], job["hash"]))
            c.execute(
                "INSERT INTO history (id, client_id, tool_type, title, content_hash, result, model, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (item_id, job["client_id"], job["tool_type"], title, job["hash"], response.text, used_model, created_at)
            )
            item = {"id": item_id, "tool_type": job["tool_type"], "title": title, "model": used_model, "created_at": created_at, "result": response.text}

        with jobs_lock:
            job.update(status="done", item=item, finished_at=time.time())
    except Exception as e:
        log.exception("Analisis gagal")
        with jobs_lock:
            job.update(status="error", error=friendly_error(e), finished_at=time.time())
    finally:
        cleanup(local_paths, remote_files)
        with jobs_lock:
            job["slots"] = None

def worker_loop():
    while True:
        job_id = job_queue.get()
        try: run_job(job_id)
        except Exception: pass
        finally: job_queue.task_done()

threading.Thread(target=worker_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# Routes Flask
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_documents():
    if not api_key:
        return jsonify({"error": "Gemini API Key belum dikonfigurasi di file .env"}), 500
    
    tool_type = request.form.get('tool_type')
    if tool_type not in TOOL_TYPES:
        return jsonify({"error": "Jenis tool tidak valid"}), 400

    force = request.form.get('force') == '1'
    saved_paths = []
    handed_off = False

    try:
        slots = prepare_slots_from_request(tool_type, saved_paths)
        present = {s["index"] for s in slots}
        if not set(MIN_SLOT_REQUIRED) <= present:
            raise ValueError("Dokumen 1 dan Dokumen 2 wajib diisi.")

        content_hash = compute_hash(tool_type, slots)

        if not force:
            with db() as c:
                row = c.execute(
                    "SELECT id, tool_type, title, model, created_at, result FROM history WHERE client_id=? AND content_hash=? ORDER BY created_at DESC LIMIT 1",
                    (g.client_id, content_hash)
                ).fetchone()
            if row:
                d = dict(row)
                d["cached"] = True
                return jsonify(d), 200

        job_id, position = enqueue_job(g.client_id, tool_type, slots, content_hash)
        handed_off = True
        return jsonify({"job_id": job_id, "position": position}), 202

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except queue.Full:
        return jsonify({"error": "Antrean sedang penuh."}), 503
    except Exception:
        log.exception("Gagal menerima permintaan analisis")
        return jsonify({"error": "Terjadi kesalahan di server."}), 500
    finally:
        if not handed_off:
            cleanup(saved_paths, [])

@app.route('/api/jobs/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job["client_id"] != g.client_id:
            return jsonify({"error": "Tugas tidak ditemukan"}), 404
        data = {"status": job["status"]}
        if job["status"] == "queued":
            idx = waiting.index(job_id) if job_id in waiting else 0
            data["position"] = idx + 1
            data["total_waiting"] = len(waiting)
        elif job["status"] == "done":
            data["item"] = job["item"]
        elif job["status"] == "error":
            data["error"] = job["error"]
    return jsonify(data)

@app.route('/api/history', methods=['GET'])
def history_list():
    with db() as c:
        rows = c.execute("SELECT id, tool_type, title, model, created_at FROM history WHERE client_id=? ORDER BY created_at DESC LIMIT 200", (g.client_id,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/history/<hid>', methods=['GET'])
def history_get(hid):
    with db() as c:
        row = c.execute("SELECT id, tool_type, title, model, created_at, result FROM history WHERE id=? AND client_id=?", (hid, g.client_id)).fetchone()
    if not row: return jsonify({"error": "Riwayat tidak ditemukan"}), 404
    return jsonify(dict(row))

@app.route('/api/history/<hid>', methods=['DELETE'])
def history_delete(hid):
    with db() as c:
        cur = c.execute("DELETE FROM history WHERE id=? AND client_id=?", (hid, g.client_id))
    if not cur.rowcount: return jsonify({"error": "Riwayat tidak ditemukan"}), 404
    return jsonify({"ok": True})

@app.route('/api/history', methods=['DELETE'])
def history_delete_all():
    with db() as c:
        cur = c.execute("DELETE FROM history WHERE client_id=?", (g.client_id,))
    return jsonify({"ok": True, "deleted": cur.rowcount})

if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=5000)