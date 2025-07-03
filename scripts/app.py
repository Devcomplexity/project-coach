# scripts/app.py

import os
import uuid
import datetime
import logging
import wikipedia
import cohere

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from duckduckgo_search.duckduckgo_search import DuckDuckGoSearchException

# ─── CONFIG ─────────────────────────────────────────────────────────────
COHERE_KEY   = os.getenv("COHERE_API_KEY")
if not COHERE_KEY:
    raise RuntimeError("Set COHERE_API_KEY as an environment variable")

co           = cohere.Client(COHERE_KEY)
MODEL        = "command-xlarge-nightly"
WEB_RESULTS  = 3
LOG_FILE     = "research.log"

# simple in-memory session store: session_id → list of "Q:…, A:…"
sessions = {}

# ─── LOGGER ─────────────────────────────────────────────────────────────
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

# ─── SCHEMAS ────────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    text: str = Field(..., description="Your question or follow-up")
    session_id: str = Field(None, description="Opaque session ID for follow-ups")

class ResetRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to clear")

# ─── SEARCH UTILITIES ──────────────────────────────────────────────────
def ddg_search(query: str):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_RESULTS):
                b = r.get("body","").strip()
                if b: out.append(b)
    except DuckDuckGoSearchException:
        logger.warning("DuckDuckGo failed")
    return out

def wiki_search(query: str):
    out = []
    try:
        for t in wikipedia.search(query, results=WEB_RESULTS):
            out.append(wikipedia.summary(t, sentences=2))
    except Exception:
        logger.warning("Wikipedia failed")
    return out

# ─── COHERE CHAT CALL ──────────────────────────────────────────────────
def call_cohere(prompt: str) -> str:
    resp = co.chat(
        model=MODEL,
        message=prompt,
        max_tokens=1024,
        temperature=0.5
    )
    if hasattr(resp, "generations") and resp.generations:
        text = resp.generations[0].text
    elif hasattr(resp, "message"):
        text = resp.message
    elif hasattr(resp, "text"):
        text = resp.text
    else:
        raise HTTPException(500, f"Unexpected Cohere response: {resp}")
    return text.strip() or "(no text returned)"

# ─── LESSON / DIALOGUE LOGIC ────────────────────────────────────────────
def research_steps(question: str, sid: str) -> (str, str):
    """
    Returns (lesson_text, session_id).  If sid is None, creates a new session.
    Otherwise uses sessions[sid] to build follow-up prompts.
    """
    is_followup = sid in sessions
    if not sid:
        sid = uuid.uuid4().hex

    if not is_followup:
        # first question: do a normal lesson
        snippets = ddg_search(question) or wiki_search(question)
        prompt = ""
        if snippets:
            prompt += "Reference snippets:\n" + "\n".join(
                f"{i+1}. {s}" for i,s in enumerate(snippets)
            ) + "\n\n"
        prompt += (
            f"You are a seasoned teacher explaining step by step how to {question}.\n"
            "Begin with a concise introduction and context.\n"
            "Then give a clear sequence of numbered steps with simple examples or analogies.\n"
            "Finish with a brief summary.\n\nLesson:\n"
        )
        lesson = call_cohere(prompt)
        sessions[sid] = [f"Q: {question}", f"A: {lesson}"]

    else:
        # follow-up: stitch context + new question
        history = "\n".join(sessions[sid])
        prompt = (
            "We have covered:\n" + history +
            f"\n\nNow the student asks a follow-up: {question}\n"
            "Answer this clearly as a teacher, building on the previous lesson.\n\nResponse:\n"
        )
        lesson = call_cohere(prompt)
        sessions[sid].append(f"Q: {question}")
        sessions[sid].append(f"A: {lesson}")

    # log
    logger.info(f"\n--- SESSION {sid} @ {datetime.datetime.utcnow().isoformat()}Z ---"
                f"\nQuestion: {question}\nLesson:\n{lesson}\n")
    return lesson, sid

# ─── FASTAPI APP ────────────────────────────────────────────────────────
app = FastAPI(title="How-To Teacher with Follow-Ups")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return HTMLResponse("""
<!doctype html><html><head><meta charset="utf-8"><title>How-To Teacher</title>
<style>body{font-family:sans-serif;margin:2rem;}textarea{width:100%;margin-bottom:.5rem;}#text{height:4rem;}#out{width:100%;height:10rem;}button{padding:.5rem;margin:.3rem 0;}</style>
</head><body>
  <h1>How-To Teacher</h1>
  <textarea id="text" placeholder="Type ‘how to…’ or ask a follow-up"></textarea><br>
  <button onclick="run()">Ask</button>
  <button onclick="reset()">New Lesson</button>
  <textarea id="out" placeholder="Your lesson..." readonly></textarea>
  <script>
    async function run(){
      let sid = localStorage.getItem("session_id");
      const body = { text: document.getElementById("text").value };
      if(sid) body.session_id = sid;
      const res = await fetch("/process", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if(res.ok){
        document.getElementById("out").value = data.result;
        localStorage.setItem("session_id", data.session_id);
      } else {
        document.getElementById("out").value = `Error ${res.status}\\n${data.detail||data}`;
      }
    }
    async function reset(){
      const sid = localStorage.getItem("session_id");
      if(sid){
        await fetch("/reset",{
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({session_id:sid})
        });
        localStorage.removeItem("session_id");
      }
      document.getElementById("out").value = "";
      alert("Session cleared. Start a new lesson.");
    }
  </script>
</body></html>
""")

@app.post("/process")
def process(req: ResearchRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty question")
    lesson, sid = research_steps(req.text.strip(), req.session_id)
    return {"result": lesson, "session_id": sid}

@app.post("/reset")
def reset(req: ResetRequest):
    sessions.pop(req.session_id, None)
    return {"reset": True}

@app.get("/logs", summary="Download logs")
def download_logs():
    if not os.path.isfile(LOG_FILE):
        raise HTTPException(404, "No logs")
    return FileResponse(LOG_FILE, media_type="text/plain", filename=LOG_FILE)
