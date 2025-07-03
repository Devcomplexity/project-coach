# scripts/app.py

import os
import datetime
import logging
import re
import wikipedia
import cohere

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from duckduckgo_search.duckduckgo_search import DuckDuckGoSearchException

# ─── CONFIG ─────────────────────────────────────────────────────────────
COHERE_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_KEY:
    raise RuntimeError("Set COHERE_API_KEY as an environment variable")

co = cohere.Client(COHERE_KEY)
MODEL = "command-xlarge-nightly"
WEB_RESULTS = 3
LOG_FILE = "research.log"

# ─── LOGGER ─────────────────────────────────────────────────────────────
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

# ─── REQUEST SCHEMA ─────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    text: str = Field(..., description="Ask your how-to question here")

# ─── HELPERS ────────────────────────────────────────────────────────────
def strip_md_heading(text: str) -> str:
    """
    Remove any leading Markdown heading (e.g. **Lesson: …**, #, ##) from the top.
    """
    lines = text.splitlines()
    while lines and re.match(r'\s*(\*{2}Lesson:|\#{1,6})', lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()

# ─── SEARCH UTILITIES ──────────────────────────────────────────────────
def ddg_search(query: str):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_RESULTS):
                body = r.get("body", "").strip()
                if body:
                    results.append(body)
    except DuckDuckGoSearchException as e:
        logger.warning(f"DuckDuckGo failed: {e}")
    return results

def wiki_search(query: str):
    results = []
    try:
        for title in wikipedia.search(query, results=WEB_RESULTS):
            summary = wikipedia.summary(title, sentences=2)
            results.append(summary)
    except Exception as e:
        logger.warning(f"Wikipedia failed: {e}")
    return results

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
        raise HTTPException(500, f"Unexpected Cohere response shape: {resp}")
    return text.strip()

# ─── PROMPT ASSEMBLY ───────────────────────────────────────────────────
def research_steps(question: str) -> str:
    snippets = ddg_search(question) or wiki_search(question)

    prompt = ""
    if snippets:
        prompt += "Reference snippets:\n" + "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(snippets)
        ) + "\n\n"

    prompt += (
        f"You are a seasoned teacher explaining step by step how to {question}.\n"
        "Begin with a concise introduction explaining the goal and context.\n"
        "Then provide a clear sequence of numbered steps, using simple examples or analogies.\n"
        "Finish with a brief summary of the key ideas.\n\n"
        "Lesson:\n"
    )

    logger.info(f"PROMPT:\n{prompt}\n{'-'*40}")
    raw = call_cohere(prompt)
    lesson = strip_md_heading(raw)
    logger.info(f"OUTPUT CLEANED:\n{lesson}\n{'='*60}")
    return lesson or "(no text returned)"

# ─── FASTAPI SETUP ─────────────────────────────────────────────────────
app = FastAPI(
    title="How-To Teacher",
    description="DuckDuckGo → Wikipedia → Cohere chat"
)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return HTMLResponse("""
<!doctype html><html><head><meta charset="utf-8"><title>How-To Teacher</title>
<style>body{font-family:sans-serif;margin:2rem;}textarea{width:100%;margin-bottom:.5rem;}#text{height:8rem;}#out{width:100%;height:12rem;}button{padding:.5rem;margin-right:.5rem;}</style>
</head><body><h1>How-To Teacher</h1>
<textarea id="text" placeholder="Ask ‘how to…’ here"></textarea>
<div><button onclick="run()">Run</button><button onclick="copy()">Copy</button></div>
<textarea id="out" placeholder="Your lesson..." readonly></textarea>
<script>
async function run(){
  const text = document.getElementById('text').value;
  const res = await fetch('/process', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text})
  });
  const data = await res.json();
  document.getElementById('out').value = res.ok ? data.result : `Error ${res.status}\\n${data.detail||data}`;
}
function copy(){navigator.clipboard.writeText(document.getElementById('out').value); alert('Copied!');}
</script></body></html>
""")

@app.post("/process")
def process(req: ResearchRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty question")
    return {"result": research_steps(req.text.strip())}

@app.get("/logs", summary="Download logs")
def download_logs():
    if not os.path.isfile(LOG_FILE):
        raise HTTPException(404, "No logs")
    return FileResponse(LOG_FILE, media_type="text/plain", filename=LOG_FILE)

@app.get("/logs/raw", response_class=HTMLResponse)
def view_logs():
    if not os.path.isfile(LOG_FILE):
        raise HTTPException(404, "No logs")
    return HTMLResponse(open(LOG_FILE, encoding="utf-8").read())
