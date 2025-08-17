from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright
import re
from typing import List, Optional

app = FastAPI(title="Komodo Transcript Fetcher (Chunked)")

# ====== Request models ======
class Req(BaseModel):
    url: HttpUrl

class MetaReq(BaseModel):
    url: HttpUrl
    max_chars: Optional[int] = 4500    # safe target per chunk
    max_segments: Optional[int] = 250  # cap lines per chunk

class ChunkReq(BaseModel):
    url: HttpUrl
    chunk_index: int                   # 0-based
    max_chars: Optional[int] = 4500
    max_segments: Optional[int] = 250

# ====== Health ======
@app.get("/")
def health():
    return {"status": "ok"}

# ====== Scraping helpers ======
TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))',
]

async def extract_transcript(page):
    # Try to open "Transcript" tab if present
    for locator in ["text=Transcript", "role=tab[name='Transcript']"]:
        if await page.locator(locator).count():
            await page.click(locator)
            break

    for sel in TRANSCRIPT_SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=7000)
            text = await page.locator(sel).inner_text()
            if text and len(text.strip()) > 10:
                return text
        except Exception:
            pass

    # Fallback: page text
    return await page.locator("body").inner_text()

def normalize_lines_to_segments(text: str):
    """
    Turn raw transcript into [{'start': seconds?, 'text': '...'}, ...]
    Accepts lines like:
      [00:23] Speaker: text
      00:23 text
      0:23.45 text
    """
    segments = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        m = re.match(r'^\[?(\d{1,2}:\d{2}(?:\.\d{1,2})?)\]?\s*(.*)$', ln)
        if m:
            ts, rest = m.groups()
            mm, ss = ts.split(":")
            try:
                sec = int(mm) * 60 + float(ss)
            except ValueError:
                sec = None
            segments.append({"start": sec, "text": (rest or "").strip()})
        else:
            segments.append({"text": ln})
    return segments

# ====== Chunking helpers ======
def chunk_by_limits(segments: List[dict], max_chars=4500, max_segments=250):
    """
    Greedy chunking by char budget & segment count.
    Returns: list[list[segment]]
    """
    chunks, current, chars = [], [], 0
    for seg in segments:
        t = seg.get("text", "")
        # Start a new chunk if we'd exceed limits
        if (len(current) >= max_segments) or (chars + len(t) > max_chars and current):
            chunks.append(current)
            current, chars = [], 0
        current.append(seg)
        chars += len(t)
    if current:
        chunks.append(current)
    return chunks

def flatten_text(segments: List[dict]):
    """
    Produce a compact plain-text block the model can read easily.
    Include timestamps if present as [mm:ss].
    """
    lines = []
    for s in segments:
        txt = s.get("text", "").strip()
        if not txt:
            continue
        start = s.get("start")
        if start is not None:
            mm = int(start // 60)
            ss = int(start % 60)
            ts = f"{mm:02d}:{ss:02d}"
            lines.append(f"[{ts}] {txt}")
        else:
            lines.append(txt)
    return "\n".join(lines)

async def load_title_and_segments(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if "tab=transcript" not in url:
                joiner = "&" if "?" in url else "?"
                await page.goto(url + f"{joiner}tab=transcript", wait_until="domcontentloaded", timeout=15000)
            raw = await extract_transcript(page)
            title = await page.title()
        finally:
            await browser.close()

    if not raw or len(raw.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")

    segs = normalize_lines_to_segments(raw)
    return title, segs

# ====== Endpoints ======

# Backward-compatible: full transcript (OK for short videos)
@app.post("/api/fetch-transcript")
async def fetch_transcript(req: Req):
    title, segs = await load_title_and_segments(str(req.url))
    return {"title": title, "transcript": segs}

# New: meta (how many chunks we'll have)
@app.post("/api/fetch-meta")
async def fetch_meta(req: MetaReq):
    title, segs = await load_title_and_segments(str(req.url))
    chunks = chunk_by_limits(segs, max_chars=req.max_chars or 4500, max_segments=req.max_segments or 250)
    total_chars = sum(len(s.get("text", "")) for s in segs)
    return {
        "title": title,
        "total_segments": len(segs),
        "estimated_total_chars": total_chars,
        "chunks_count": len(chunks),
        "max_chars": req.max_chars or 4500,
        "max_segments": req.max_segments or 250
    }

# New: fetch a single chunk (safe size for GPT Action)
@app.post("/api/fetch-chunk")
async def fetch_chunk(req: ChunkReq):
    if req.chunk_index < 0:
        raise HTTPException(status_code=400, detail="chunk_index must be >= 0")

    title, segs = await load_title_and_segments(str(req.url))
    chunks = chunk_by_limits(segs, max_chars=req.max_chars or 4500, max_segments=req.max_segments or 250)

    if req.chunk_index >= len(chunks):
        raise HTTPException(status_code=416, detail="chunk_index out of range")

    chunk = chunks[req.chunk_index]
    text_block = flatten_text(chunk)
    start_idx = sum(len(c) for c in chunks[:req.chunk_index])
    end_idx = start_idx + len(chunk) - 1

    return {
        "title": title,
        "chunk_index": req.chunk_index,
        "chunks_count": len(chunks),
        "segments_in_chunk": len(chunk),
        "segments_range": [start_idx, end_idx],
        "text": text_block,
        "segments": chunk
    }
