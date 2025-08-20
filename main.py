import os
import re
import math
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Iterable, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, HttpUrl
from bs4 import BeautifulSoup

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

# =========================
# App setup
# =========================
app = FastAPI(title="Komodo Transcript Service (Chunked)", version="1.2.0")

# CORS (tighten in prod as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# =========================
# Render/Playwright tuning
# =========================
MAX_PW_CONTEXTS = int(os.getenv("MAX_PW_CONTEXTS", "2"))    # safe for 512 MB
PW_NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "30000"))

# Auto-scroll to hydrate lazy transcript content inside the scroll container
AUTO_SCROLL_MAX_STEPS = int(os.getenv("AUTO_SCROLL_MAX_STEPS", "20"))
AUTO_SCROLL_SLEEP_MS = int(os.getenv("AUTO_SCROLL_SLEEP_MS", "200"))

_browser: Optional[Browser] = None
_pw = None
_pw_sema = asyncio.Semaphore(MAX_PW_CONTEXTS)

@app.on_event("startup")
async def _startup_playwright():
    global _pw, _browser
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )

@app.on_event("shutdown")
async def _shutdown_playwright():
    global _pw, _browser
    try:
        if _browser:
            await _browser.close()
    finally:
        _browser = None
        if _pw:
            await _pw.stop()
            _pw = None

@asynccontextmanager
async def context_page():
    """
    Create a lightweight isolated context+page using the shared browser,
    with concurrency capped by a semaphore.
    """
    if _browser is None:
        raise HTTPException(status_code=500, detail="Browser not initialized.")
    async with _pw_sema:
        context: BrowserContext = await _browser.new_context()
        page: Page = await context.new_page()
        try:
            yield context, page
        finally:
            await context.close()

# =========================
# Models per API contract
# =========================
class FetchMetaBody(BaseModel):
    url: HttpUrl
    max_chars: int = Field(4500, ge=1)
    max_segments: int = Field(250, ge=1)

class FetchMetaReply(BaseModel):
    title: Optional[str] = None
    total_segments: int
    estimated_total_chars: int
    chunks_count: int
    max_chars: int
    max_segments: int

class FetchChunkBody(BaseModel):
    url: HttpUrl
    chunk_index: int = Field(..., ge=0)
    max_chars: int = Field(4500, ge=1)
    max_segments: int = Field(250, ge=1)

class SegmentOut(BaseModel):
    start: Optional[float] = None  # seconds
    text: str

class FetchChunkReply(BaseModel):
    title: Optional[str] = None
    chunk_index: int
    chunks_count: int
    segments_in_chunk: int
    segments_range: List[int]  # [start_idx, end_idx_inclusive]
    text: str
    segments: List[SegmentOut]

# =========================
# Navigation & parsing
# =========================
async def safe_goto(page: Page, url: str, timeout_ms: int = PW_NAV_TIMEOUT_MS) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeoutError as e:
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms // 2)
        except Exception as e2:
            raise HTTPException(status_code=504, detail=f"Timeout navigating to {url}: {e2}") from e

async def _auto_scroll_transcript(page: Page) -> None:
    """
    Scroll the transcript container to force lazy content to render.
    Falls back to window scroll if the pane isn't found.
    """
    # Based on provided HTML: scrollable transcript pane
    pane_sel = '.overflow-auto.w-full.h-full'
    try:
        if await page.locator(pane_sel).count() > 0:
            pane = page.locator(pane_sel).first
            last_h = -1
            for _ in range(AUTO_SCROLL_MAX_STEPS):
                h = await pane.evaluate('(el) => el.scrollHeight')
                if h == last_h:
                    break
                last_h = h
                await pane.evaluate('(el) => el.scrollTo(0, el.scrollHeight)')
                await page.wait_for_timeout(AUTO_SCROLL_SLEEP_MS)
        else:
            # Fallback: scroll the window
            last_h = -1
            for _ in range(AUTO_SCROLL_MAX_STEPS):
                h = await page.evaluate('document.documentElement.scrollHeight')
                if h == last_h:
                    break
                last_h = h
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await page.wait_for_timeout(AUTO_SCROLL_SLEEP_MS)
    except Exception:
        # Non-fatal; continue with whatever is loaded
        pass

async def fetch_transcript_html(url: str) -> str:
    """
    Open the Komodo recording URL on the transcript tab, hydrate all rows by scrolling,
    then return the page HTML.
    """
    joiner = "&" if "?" in url else "?"
    transcript_url = url + f"{joiner}tab=transcript"
    async with context_page() as (_ctx, page):
        await safe_goto(page, transcript_url)
        # Hydrate lazy content inside the transcript pane
        await _auto_scroll_transcript(page)
        html = await page.content()
        return html

# Timestamp regex: hh?:mm:ss or m:ss
_TS_RE = re.compile(r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b")

def _parse_hhmmss_to_seconds(ts: str) -> Optional[float]:
    m = _TS_RE.fullmatch(ts.strip())
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    return float(hh * 3600 + mm * 60 + ss)

def extract_segments(html: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Extract (title, segments[{start, text}]) from Komodo transcript DOM.
    Targets grid rows with a timestamp anchor, then falls back gracefully.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    segments: List[Dict[str, Any]] = []

    # Primary selector: grid rows that hold [timestamp] [text]
    # Example class combo seen: div.grid.gap-2.p-2.rounded.text-base (with grid-template-columns: 4.5ch 1fr)
    row_selector = 'div.grid.gap-2.p-2.rounded.text-base'
    rows = soup.select(row_selector)

    for row in rows:
        # first column: timestamp anchor like <a href="... ?t=...">00:05</a>
        ts_a = row.select_one('span > a[href*="?t="], a[href*="?t="]')
        ts_text = ts_a.get_text(strip=True) if ts_a else ""
        start = None
        if ts_text:
            parts = ts_text.split(":")
            # normalize m:ss -> 0:m:ss for parser
            if len(parts) == 2:
                ts_norm = f"0:{ts_text}"
            else:
                ts_norm = ts_text
            start = _parse_hhmmss_to_seconds(ts_norm)

        # second column: tokenized words inside flex container → join strings
        # Heuristic: take the largest text block that is not the timestamp cell
        # Prefer the sibling element after the first span (timestamp)
        # If not found, fall back to row's full text minus timestamp
        second_col = None

        # Try: find direct children; often the second column is a div/span sibling
        children = [c for c in row.children if getattr(c, "name", None)]
        if len(children) >= 2:
            second_col = children[1]
        else:
            # fallback: pick the largest texty descendant
            cand = None
            cand_len = -1
            for d in row.find_all(True):
                t = d.get_text(strip=True)
                l = len(t)
                if l > cand_len:
                    cand_len = l
                    cand = d
            second_col = cand or row

        line_text = " ".join(second_col.stripped_strings).strip()
        # Remove timestamp echo if it leaked into text
        if ts_text and line_text.startswith(ts_text):
            line_text = line_text[len(ts_text):].strip()
        line_text = re.sub(r"\s+", " ", line_text)

        if line_text:
            segments.append({"start": start, "text": line_text})

    # Fallback 1: use any anchor with ?t= as row head, grab its container text
    if not segments:
        for a in soup.select('a[href*="?t="]'):
            ts = a.get_text(strip=True)
            container = a.find_parent('div', class_='grid') or a.parent
            text_block = " ".join(container.stripped_strings) if container else ts
            # Remove the timestamp token from the beginning, if present
            if text_block.startswith(ts):
                text_block = text_block[len(ts):].strip()
            text_block = re.sub(r"\s+", " ", text_block)
            if text_block:
                # parse ts for start
                parts = ts.split(":")
                ts_norm = f"0:{ts}" if len(parts) == 2 else ts
                start = _parse_hhmmss_to_seconds(ts_norm) if ts else None
                segments.append({"start": start, "text": text_block})

    # Fallback 2: whole-page lines (last resort)
    if not segments:
        body_text = soup.get_text(separator="\n", strip=True)
        for ln in (ln.strip() for ln in body_text.splitlines() if ln.strip()):
            segments.append({"start": None, "text": re.sub(r"\s+", " ", ln)})

    return title, segments

# =========================
# Chunking over segments
# =========================
def compute_chunks_boundaries(
    segments: List[Dict[str, Any]],
    max_chars: int,
    max_segments: int,
) -> List[Tuple[int, int]]:
    """
    Produce a list of (start_idx, end_idx_exclusive) chunk boundaries over 'segments',
    respecting both max_chars and max_segments.
    """
    bounds: List[Tuple[int, int]] = []
    n = len(segments)
    i = 0
    while i < n:
        used_chars = 0
        segs = 0
        j = i
        while j < n:
            seg_text = segments[j]["text"]
            seg_len = len(seg_text)
            if segs + 1 > max_segments or (used_chars + seg_len) > max_chars:
                break
            used_chars += seg_len
            segs += 1
            j += 1
        if j == i:
            # single segment too large → hard cut to avoid infinite loop
            take_text = segments[j]["text"][:max_chars]
            segments[j]["text"] = take_text  # mutate in place for consistency
            j += 1
        bounds.append((i, j))
        i = j
    return bounds

def join_text(segments: List[Dict[str, Any]], start: int, end: int) -> str:
    return "\n".join(seg["text"] for seg in segments[start:end])

# =========================
# Minimal ops endpoints
# =========================
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    # For HEAD, FastAPI ignores the body; status 200 still returned.
    return JSONResponse({
        "service": "Komodo Transcript Fetcher (Chunked)",
        "status": "ok",
        "health": "/health",
        "actions": {
            "fetchMeta": "POST /api/fetch-meta",
            "fetchChunk": "POST /api/fetch-chunk",
        }
    })

@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
async def health():
    return {"status": "ok", "browser_initialized": _browser is not None}

# =========================
# ACTION: POST /api/fetch-meta
# =========================
@app.post("/api/fetch-meta", response_model=FetchMetaReply)
async def fetch_meta(body: FetchMetaBody):
    """
    Get transcript metadata and chunk count for the given URL and limits.
    """
    html = await fetch_transcript_html(str(body.url))
    title, segments = extract_segments(html)

    total_segments = len(segments)
    estimated_total_chars = sum(len(s["text"]) for s in segments)

    boundaries = compute_chunks_boundaries(segments, body.max_chars, body.max_segments)
    chunks_count = len(boundaries)

    return FetchMetaReply(
        title=title,
        total_segments=total_segments,
        estimated_total_chars=estimated_total_chars,
        chunks_count=chunks_count,
        max_chars=body.max_chars,
        max_segments=body.max_segments,
    )

# =========================
# ACTION: POST /api/fetch-chunk
# =========================
@app.post("/api/fetch-chunk", response_model=FetchChunkReply, responses={416: {"description": "chunk_index out of range"}})
async def fetch_chunk(body: FetchChunkBody):
    """
    Return a specific chunk identified by chunk_index with text and segments.
    """
    html = await fetch_transcript_html(str(body.url))
    title, segments = extract_segments(html)

    # Recompute SAME boundaries using provided limits to ensure determinism
    boundaries = compute_chunks_boundaries(segments, body.max_chars, body.max_segments)
    chunks_count = len(boundaries)

    if body.chunk_index < 0 or body.chunk_index >= chunks_count:
        raise HTTPException(status_code=416, detail="chunk_index out of range")

    start, end = boundaries[body.chunk_index]
    seg_slice = segments[start:end]
    text = join_text(segments, start, end)

    segments_range = [start, end - 1] if end > start else [start, start]
    out_segments = [SegmentOut(start=s.get("start"), text=s["text"]) for s in seg_slice]

    return FetchChunkReply(
        title=title,
        chunk_index=body.chunk_index,
        chunks_count=chunks_count,
        segments_in_chunk=len(seg_slice),
        segments_range=segments_range,
        text=text,
        segments=out_segments,
    )

# =========================
# Entrypoint (Render)
# =========================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # workers=1 so we keep a single shared browser
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)
