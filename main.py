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
app = FastAPI(title="Komodo Transcript Service (Chunked)", version="1.1.0")

# CORS (tighten in prod as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# =========================
# Render/Playwright tuning
# =========================
MAX_PW_CONTEXTS = int(os.getenv("MAX_PW_CONTEXTS", "2"))  # safe for 512 MB
PW_NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "30000"))

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

async def fetch_transcript_html(url: str) -> str:
    joiner = "&" if "?" in url else "?"
    transcript_url = url + f"{joiner}tab=transcript"
    async with context_page() as (_ctx, page):
        await safe_goto(page, transcript_url)
        # brief wait for dynamic content (kept small for memory/latency)
        await page.wait_for_timeout(800)
        html = await page.content()
        return html

_TS_RE = re.compile(
    r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b"  # hh?:mm:ss or m:ss
)

def _parse_timestamp_to_seconds(text: str) -> Optional[float]:
    m = _TS_RE.search(text)
    if not m:
        return None
    h = m.group(1)
    mnt = m.group(2)
    sec = m.group(3)
    hh = int(h) if h is not None else 0
    mm = int(mnt)
    ss = int(sec)
    return float(hh * 3600 + mm * 60 + ss)

def extract_segments(html: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Attempt to extract (title, segments[{start, text}]) from Komodo transcript.
    Falls back to newline-based segmentation if no structured nodes are found.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    segments: List[Dict[str, Any]] = []

    # Heuristic 1: look for items with obvious transcript classes (common on many players)
    candidates = soup.select('[data-test="transcript-segment"], .transcript-segment, .TranscriptItem, .transcript__row')
    for node in candidates:
        txt = node.get_text(separator=" ", strip=True)
        if not txt:
            continue
        start = _parse_timestamp_to_seconds(txt)
        segments.append({"start": start, "text": txt})

    if not segments:
        # Heuristic 2: look for list-like containers with span/p tags
        rows = soup.select("li, p, div")
        approx = []
        for r in rows:
            t = r.get_text(separator=" ", strip=True)
            if t and len(t.split()) >= 3:
                approx.append(t)
        # Deduplicate while keeping order
        seen = set()
        out = []
        for t in approx:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if out:
            for t in out:
                segments.append({"start": _parse_timestamp_to_seconds(t), "text": t})

    if not segments:
        # Fallback: split by newline on full text (last resort)
        body_text = soup.get_text(separator="\n", strip=True) if soup else ""
        lines = [ln.strip() for ln in body_text.splitlines() if ln.strip()]
        for ln in lines:
            segments.append({"start": _parse_timestamp_to_seconds(ln), "text": ln})

    # Final cleanup: remove empties and collapse whitespace
    cleaned = []
    for s in segments:
        text = re.sub(r"\s+", " ", (s.get("text") or "")).strip()
        if text:
            cleaned.append({"start": s.get("start"), "text": text})
    return title, cleaned

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
            # take as much as possible from this one segment
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
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": "Komodo Transcript Fetcher (Chunked)",
        "status": "ok",
        "health": "/health",
        "actions": {
            "fetchMeta": "POST /api/fetch-meta",
            "fetchChunk": "POST /api/fetch-chunk",
        }
    })

@app.get("/health", include_in_schema=False)
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

    # Compute chunk boundaries with provided limits
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
        # 416 per contract
        raise HTTPException(status_code=416, detail="chunk_index out of range")

    start, end = boundaries[body.chunk_index]
    seg_slice = segments[start:end]
    text = join_text(segments, start, end)

    # segments_range must be inclusive end per contract
    segments_range = [start, end - 1] if end > start else [start, start]

    # shape segments to output schema
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
