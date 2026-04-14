import os
import re
import json
import sqlite3
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from starlette.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY and not GROQ_API_KEY:
    raise RuntimeError(
        "❌ Hech qanday AI API key topilmadi!\n"
        "Kamida bittasini o'rnating:\n"
        "  set GEMINI_API_KEY=your_key   (https://aistudio.google.com/apikey)\n"
        "  set GROQ_API_KEY=your_key     (https://console.groq.com)"
    )

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# LEETCODE PROBLEM CATALOGUE  (slug, title, difficulty)
# ─────────────────────────────────────────────
LEETCODE_CATALOGUE = [
    ("two-sum","Two Sum","Easy"),
    ("valid-parentheses","Valid Parentheses","Easy"),
    ("merge-two-sorted-lists","Merge Two Sorted Lists","Easy"),
    ("best-time-to-buy-and-sell-stock","Best Time to Buy and Sell Stock","Easy"),
    ("valid-palindrome","Valid Palindrome","Easy"),
    ("invert-binary-tree","Invert Binary Tree","Easy"),
    ("valid-anagram","Valid Anagram","Easy"),
    ("binary-search","Binary Search","Easy"),
    ("flood-fill","Flood Fill","Easy"),
    ("lowest-common-ancestor-of-a-binary-search-tree","Lowest Common Ancestor of a BST","Easy"),
    ("balanced-binary-tree","Balanced Binary Tree","Easy"),
    ("linked-list-cycle","Linked List Cycle","Easy"),
    ("first-bad-version","First Bad Version","Easy"),
    ("ransom-note","Ransom Note","Easy"),
    ("climbing-stairs","Climbing Stairs","Easy"),
    ("longest-common-prefix","Longest Common Prefix","Easy"),
    ("single-number","Single Number","Easy"),
    ("palindrome-linked-list","Palindrome Linked List","Easy"),
    ("move-zeroes","Move Zeroes","Easy"),
    ("missing-number","Missing Number","Easy"),
    ("contains-duplicate","Contains Duplicate","Easy"),
    ("counting-bits","Counting Bits","Easy"),
    ("same-tree","Same Tree","Easy"),
    ("reverse-linked-list","Reverse Linked List","Easy"),
    ("maximum-depth-of-binary-tree","Maximum Depth of Binary Tree","Easy"),
    ("remove-duplicates-from-sorted-array","Remove Duplicates from Sorted Array","Easy"),
    ("sqrt","Sqrt(x)","Easy"),
    ("roman-to-integer","Roman to Integer","Easy"),
    ("is-subsequence","Is Subsequence","Easy"),
    ("number-of-1-bits","Number of 1 Bits","Easy"),
    ("reverse-bits","Reverse Bits","Easy"),
    ("intersection-of-two-linked-lists","Intersection of Two Linked Lists","Easy"),
    ("majority-element","Majority Element","Easy"),
    ("happy-number","Happy Number","Easy"),
    ("summary-ranges","Summary Ranges","Easy"),
    ("power-of-two","Power of Two","Easy"),
    ("arranging-coins","Arranging Coins","Easy"),
    ("hamming-distance","Hamming Distance","Easy"),
    ("fibonacci-number","Fibonacci Number","Easy"),
    ("middle-of-the-linked-list","Middle of the Linked List","Easy"),
    ("maximum-average-subarray-i","Maximum Average Subarray I","Easy"),
    ("find-pivot-index","Find Pivot Index","Easy"),
]

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

DB_FILE = "uzleetcode.db"

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def get_db():
    """Opens a connection to the SQLite database file."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates tables if they don't exist yet. Runs once on startup."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            is_admin  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS problems (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            slug         TEXT UNIQUE NOT NULL,
            title_en     TEXT NOT NULL,
            title_uz     TEXT,
            content_uz   TEXT,
            difficulty   TEXT,
            is_published INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS submissions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            problem_slug TEXT    NOT NULL,
            code         TEXT    NOT NULL,
            is_correct   INTEGER NOT NULL,
            feedback     TEXT,
            submitted_at TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# LIFESPAN  (must be defined BEFORE app = FastAPI)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
            (ADMIN_USERNAME, ADMIN_PASSWORD)
        )
        conn.commit()
        print(f"✅ Default admin created: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    conn.close()
    print("✅ Database ready")
    yield


# ─────────────────────────────────────────────
# FASTAPI APP  (defined AFTER lifespan)
# ─────────────────────────────────────────────

app = FastAPI(title="UzLeetCode", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Kirish talab qilinadi")
    try:
        username, password = credentials.credentials.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Token noto'g'ri")

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password)
    ).fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Foydalanuvchi topilmadi")
    return dict(user)


def require_admin(user=Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin huquqi kerak")
    return user


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class SubmitRequest(BaseModel):
    problem_slug: str
    code: str

class ProblemSyncRequest(BaseModel):
    slug: str


# ─────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────


def _ask_gemini(prompt: str) -> str:
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    response = requests.post(
        url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _ask_groq(prompt: str) -> str:
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def gemini_ask(prompt: str) -> str:
    """Gemini bilan urinib ko'r, muvaffaqiyatsiz bo'lsa Groq'ga o'tadi."""
    if GEMINI_API_KEY:
        try:
            return _ask_gemini(prompt)
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429:
                print("⚠️ Gemini rate limit, Groq'ga o'tilmoqda...")
            elif code not in (500, 502, 503):
                raise HTTPException(status_code=502, detail=f"Gemini xatosi ({code}): {e.response.text[:200]}")
        except requests.exceptions.RequestException:
            print("⚠️ Gemini ulanmadi, Groq'ga o'tilmoqda...")

    if GROQ_API_KEY:
        try:
            return _ask_groq(prompt)
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            raise HTTPException(status_code=502, detail=f"Groq xatosi ({code}): {e.response.text[:200]}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Groq ulanish xatosi: {str(e)}")

    raise HTTPException(status_code=502, detail="Hech qanday AI API ishlamadi")


def translate_problem(title_en: str, content_en: str) -> dict:
    prompt = f"""Quyidagi LeetCode masalasini o'zbek tiliga tarjima qil.
BARCHA qismlarni tarjima qil: tavsif, misollar (Input/Output/Explanation), cheklovlar.
Kod bloklari ichini tarjima qilma, aynan qoldir.
"Follow-up", "Could you", "Can you come up with" kabi qismlarni TARJIMA QILMA va YOZMA — ular shart emas.
Faqat quyidagi formatda yoz, boshqa hech narsa qo'shma:

SARLAVHA: <sarlavha tarjimasi>
MAZMUN:
<to'liq mazmun tarjimasi>

Inglizcha sarlavha: {title_en}
Inglizcha mazmun:
{content_en}"""

    result = _ask_groq(prompt)
    print(f"[DEBUG] Groq raw result ({len(result)} chars):\n{result[:400]}\n---")

    title_uz = title_en
    content_lines = []
    in_content = False

    for line in result.split("\n"):
        if line.startswith("SARLAVHA:"):
            title_uz = line.replace("SARLAVHA:", "").strip()
        elif line.strip() == "MAZMUN:" or line.startswith("MAZMUN:\n"):
            in_content = True
        elif in_content:
            content_lines.append(line)

    content_uz = "\n".join(content_lines).strip()

    if not content_uz or len(content_uz) < len(content_en) * 0.4:
        print(f"[WARN] Translation too short or empty, keeping English")
        content_uz = content_en

    return {"title_uz": title_uz, "content_uz": content_uz}


def judge_code(problem_title: str, problem_content: str, user_code: str) -> dict:
    prompt = f"""
Sen dasturlash murabbiyisan. Agar xato bo'lsa, sen hint ber. Foydalanuvchi quyidagi masalani yechishga harakat qildi.

Masala: {problem_title}
Masala matni: {problem_content[:800]}

Foydalanuvchi kodi:
{user_code}

MUHIM QOIDALAR:
- Bu boshlang'ich o'quvchilar uchun. Faqat KOD TO'G'RI NATIJA BERADIMI — shuni tekshir.
- Time complexity, space complexity, O(n), O(log n) kabi talablarni TEKSHIRMA va HISOBGA OLMA.
- Kod ishlasa va masala shartini bajarsa — is_correct: true.
- Faqat mantiqiy xato bo'lsa — is_correct: false.

FAQAT quyidagi JSON formatida javob ber, boshqa hech narsa yozma:
{{
  "is_correct": true yoki false,
  "feedback": "O'zbek tilida qisqa izoh (2-3 jumla)"
}}

Boshqa hech narsa yozma. Faqat JSON.
"""
    result = gemini_ask(prompt)
    result = result.replace("```json", "").replace("```", "").strip()
    data = json.loads(result)
    return {
        "is_correct": bool(data.get("is_correct", False)),
        "feedback": data.get("feedback", "Xato yuz berdi")
    }


# ─────────────────────────────────────────────
# LEETCODE FETCH HELPER
# ─────────────────────────────────────────────

def fetch_problem_from_leetcode(slug: str) -> dict:
    query = """
    query getQuestion($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        title
        content
        difficulty
      }
    }
    """
    response = requests.post(
        "https://leetcode.com/graphql",
        json={"query": query, "variables": {"titleSlug": slug}},
        headers={"Content-Type": "application/json"},
        timeout=15
    )
    response.raise_for_status()
    data = response.json()
    q = data["data"]["question"]

    if not q:
        raise HTTPException(status_code=404, detail=f"'{slug}' LeetCode'da topilmadi")

    raw = q["content"] or ""
    print(f"[DEBUG] Raw HTML length: {len(raw)}")
    print(f"[DEBUG] Raw HTML preview:\n{raw[:500]}\n---")

    # Preserve examples: <pre> blocks extract first
    pre_blocks = re.findall(r"<pre>(.*?)</pre>", raw, re.DOTALL)

    # Replace structural tags with newlines
    raw = re.sub(r"<p>", "", raw)
    raw = re.sub(r"</p>", "\n\n", raw)
    raw = re.sub(r"<ul>|</ul>", "\n", raw)
    raw = re.sub(r"<ol>|</ol>", "\n", raw)
    raw = re.sub(r"<li>", "• ", raw)
    raw = re.sub(r"</li>", "\n", raw)
    raw = re.sub(r"<pre>.*?</pre>", lambda m: "\n```\n" + re.sub(r"<[^>]+>", "", m.group()) + "\n```\n", raw, flags=re.DOTALL)
    raw = re.sub(r"<strong>(.*?)</strong>", r"**\1**", raw)
    raw = re.sub(r"<em>(.*?)</em>", r"_\1_", raw)
    raw = re.sub(r"<sup>(.*?)</sup>", r"^\1", raw)
    raw = re.sub(r"<code>(.*?)</code>", r"`\1`", raw)
    raw = re.sub(r"<[^>]+>", "", raw)           # strip remaining tags
    raw = re.sub(r"&nbsp;", " ", raw)
    raw = re.sub(r"&lt;", "<", raw)
    raw = re.sub(r"&gt;", ">", raw)
    raw = re.sub(r"&amp;", "&", raw)
    raw = re.sub(r"&quot;", '"', raw)
    content_clean = re.sub(r"\n{3,}", "\n\n", raw).strip()

    # Follow-up va complexity talablarini olib tashlaymiz
    content_clean = re.sub(
        r"\n*(Follow[ -]up|Could you|Can you come up|Note that|What if).*",
        "", content_clean, flags=re.IGNORECASE | re.DOTALL
    ).strip()

    print(f"[DEBUG] Cleaned content length: {len(content_clean)}")
    print(f"[DEBUG] Cleaned content:\n{content_clean}\n===")

    return {
        "title": q["title"],
        "content": content_clean,
        "difficulty": q["difficulty"]
    }


# ─────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/api/signup")
def signup(body: SignupRequest):
    if len(body.username) < 3:
        raise HTTPException(status_code=400, detail="Username kamida 3 ta harf bo'lishi kerak")
    if len(body.password) < 4:
        raise HTTPException(status_code=400, detail="Parol kamida 4 ta belgi bo'lishi kerak")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (body.username, body.password)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Bu username band")
    finally:
        conn.close()

    return {"token": f"{body.username}:{body.password}", "username": body.username, "is_admin": False}


@app.post("/api/login")
def login(body: LoginRequest):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (body.username, body.password)
    ).fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Username yoki parol noto'g'ri")

    return {
        "token": f"{body.username}:{body.password}",
        "username": body.username,
        "is_admin": bool(user["is_admin"])
    }


# ─────────────────────────────────────────────
# PROBLEM ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/problems")
def list_problems():
    """Katalogdagi barcha masalalar + DB'dagi tarjima holati."""
    conn = get_db()
    db_rows = conn.execute("SELECT slug, title_uz FROM problems").fetchall()
    conn.close()
    db_map = {r["slug"]: r["title_uz"] for r in db_rows}

    seen = set()
    result = []
    for slug, title_en, difficulty in LEETCODE_CATALOGUE:
        if slug in seen:
            continue
        seen.add(slug)
        result.append({
            "slug": slug,
            "title_en": title_en,
            "title_uz": db_map.get(slug),
            "difficulty": difficulty,
        })
    return result

@app.get("/api/admin/problems")
def admin_list_problems(admin=Depends(require_admin)):
    """Admin uchun — katalog + DB holati."""
    conn = get_db()
    db_rows = conn.execute("SELECT slug, title_uz, is_published FROM problems").fetchall()
    conn.close()
    db_map = {r["slug"]: dict(r) for r in db_rows}

    seen = set()
    result = []
    for slug, title_en, difficulty in LEETCODE_CATALOGUE:
        if slug in seen:
            continue
        seen.add(slug)
        db = db_map.get(slug, {})
        result.append({
            "slug": slug,
            "title_en": title_en,
            "title_uz": db.get("title_uz"),
            "difficulty": difficulty,
            "translated": bool(db.get("title_uz")),
        })
    return result


@app.get("/api/problems/{slug}")
def get_problem(slug: str, user=Depends(get_current_user)):
    # Katalogda borligini tekshir
    catalogue_entry = next((x for x in LEETCODE_CATALOGUE if x[0] == slug), None)
    if not catalogue_entry:
        raise HTTPException(status_code=404, detail="Masala katalogda topilmadi")

    conn = get_db()
    problem = conn.execute("SELECT * FROM problems WHERE slug=?", (slug,)).fetchone()

    # Agar bazada bor va tarjima qilingan bo'lsa — qaytaramiz
    if problem and problem["content_uz"]:
        result = dict(problem)
        conn.close()
        return result

    # LeetCode'dan yuklab tarjima qilamiz
    lc_data = fetch_problem_from_leetcode(slug)
    translated = translate_problem(lc_data["title"], lc_data["content"])


    if problem:
        # Bazada bor lekin tarjima yo'q — update
        conn.execute(
            "UPDATE problems SET title_uz=?, content_uz=? WHERE slug=?",
            (translated["title_uz"], translated["content_uz"], slug)
        )
    else:
        # Bazada yo'q — insert
        conn.execute(
            "INSERT INTO problems (slug, title_en, title_uz, content_uz, difficulty) VALUES (?,?,?,?,?)",
            (slug, lc_data["title"], translated["title_uz"], translated["content_uz"], lc_data["difficulty"])
        )
    conn.commit()

    result = {
        "slug": slug,
        "title_en": lc_data["title"],
        "title_uz": translated["title_uz"],
        "content_uz": translated["content_uz"],
        "difficulty": lc_data["difficulty"],
    }
    conn.close()
    return result


# ─────────────────────────────────────────────
# SUBMISSION ENDPOINT
# ─────────────────────────────────────────────

@app.post("/api/submit")
def submit_code(body: SubmitRequest, user=Depends(get_current_user)):
    conn = get_db()
    problem = conn.execute(
        "SELECT * FROM problems WHERE slug = ?", (body.problem_slug,)
    ).fetchone()

    if not problem:
        conn.close()
        raise HTTPException(status_code=404, detail="Masala topilmadi")

    problem = dict(problem)
    result = judge_code(
        problem_title=problem["title_uz"] or problem["title_en"],
        problem_content=problem.get("content_uz", ""),
        user_code=body.code
    )

    conn.execute(
        """INSERT INTO submissions (user_id, problem_slug, code, is_correct, feedback)
           VALUES (?, ?, ?, ?, ?)""",
        (user["id"], body.problem_slug, body.code, int(result["is_correct"]), result["feedback"])
    )
    conn.commit()
    conn.close()

    return {
        "is_correct": result["is_correct"],
        "feedback": result["feedback"]
    }


# ─────────────────────────────────────────────
# PROFILE ENDPOINT
# ─────────────────────────────────────────────

@app.get("/api/profile")
def get_profile(user=Depends(get_current_user)):
    conn = get_db()
    user_id = user["id"]

    total = conn.execute(
        "SELECT COUNT(*) FROM submissions WHERE user_id = ?", (user_id,)
    ).fetchone()[0]

    correct = conn.execute(
        "SELECT COUNT(*) FROM submissions WHERE user_id = ? AND is_correct = 1", (user_id,)
    ).fetchone()[0]

    recent = conn.execute(
        """SELECT s.problem_slug, s.is_correct, s.feedback, s.submitted_at, p.title_uz
           FROM submissions s
           LEFT JOIN problems p ON s.problem_slug = p.slug
           WHERE s.user_id = ?
           ORDER BY s.submitted_at DESC
           LIMIT 10""",
        (user_id,)
    ).fetchall()

    conn.close()
    return {
        "username": user["username"],
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "recent": [dict(r) for r in recent]
    }


# ─────────────────────────────────────────────
# ADMIN ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/admin/stats")
def admin_stats(admin=Depends(require_admin)):
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_admin = 0").fetchone()[0]
    total_problems = conn.execute("SELECT COUNT(*) FROM problems").fetchone()[0]
    total_submissions = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    correct_submissions = conn.execute("SELECT COUNT(*) FROM submissions WHERE is_correct = 1").fetchone()[0]

    top_problems = conn.execute(
        """SELECT problem_slug, COUNT(*) as attempts
           FROM submissions GROUP BY problem_slug
           ORDER BY attempts DESC LIMIT 5"""
    ).fetchall()

    recent_users = conn.execute(
        "SELECT username FROM users WHERE is_admin = 0 ORDER BY id DESC LIMIT 5"
    ).fetchall()

    conn.close()
    return {
        "total_users": total_users,
        "total_problems": total_problems,
        "total_submissions": total_submissions,
        "correct_submissions": correct_submissions,
        "accuracy_percent": round((correct_submissions / total_submissions * 100) if total_submissions else 0, 1),
        "top_problems": [dict(r) for r in top_problems],
        "recent_users": [r["username"] for r in recent_users]
    }


@app.post("/api/admin/add-problem")
def add_problem(body: ProblemSyncRequest, admin=Depends(require_admin)):
    lc_data = fetch_problem_from_leetcode(body.slug)
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO problems (slug, title_en, difficulty) VALUES (?, ?, ?)",
            (body.slug, lc_data["title"], lc_data["difficulty"])
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Bu masala allaqachon mavjud")
    conn.close()
    return {"message": f"'{lc_data['title']}' masalasi qo'shildi", "slug": body.slug}


@app.delete("/api/admin/delete-problem/{slug}")
def delete_problem(slug: str, admin=Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM problems WHERE slug = ?", (slug,))
    conn.commit()
    conn.close()
    return {"message": f"'{slug}' o'chirildi"}


@app.post("/api/admin/create-admin")
def create_admin(body: SignupRequest, admin=Depends(require_admin)):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
            (body.username, body.password)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Bu username band")
    finally:
        conn.close()
    return {"message": f"Admin '{body.username}' yaratildi"}


# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())