import os
import re
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Dict, Any

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
app = FastAPI(title="Komodo Transcript Service (Chunked)", version="1.3.0")

# CORS (tighten in prod as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# =========================
# Render/Playwright tuning
# =========================
MAX_PW_CONTEXTS = int(os.getenv("MAX_PW_CONTEXTS", "2"))    # safe for 512 MB
PW_NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "30000"))

# Hydration knobs
AUTO_SCROLL_MAX_STEPS = int(os.getenv("AUTO_SCROLL_MAX_STEPS", "60"))   # more steps for longer videos
AUTO_SCROLL_SLEEP_MS = int(os.getenv("AUTO_SCROLL_SLEEP_MS", "150"))    # small delay between scrolls
HYDRATE_STABLE_ROUNDS = int(os.getenv("HYDRATE_STABLE_ROUNDS", "3"))    # require N unchanged counts before stopping
TRANSCRIPT_PANE_SELECTORS = os.getenv(
    "TRANSCRIPT_PANE_SELECTORS",
    # comma-separated list; can be adjusted via env without code changes
    '[data-test="transcript"],.overflow-auto.w-full.h-full,.transcript,.Transcript,.transcript-container,.TranscriptContainer'
).split(",")

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
# Navigation & hydration
# =========================
async def safe_goto(page: Page, url: str, timeout_ms: int = PW_NAV_TIMEOUT_MS) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeoutError as e:
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms // 2)
        except Exception as e2:
            raise HTTPException(status_code=504, detail=f"Timeout navigating to {url}: {e2}") from e

async def _count_transcript_anchors(page: Page) -> int:
    # anchors that link with ?t=timestamp are reliable transcript markers
    return await page.evaluate('document.querySelectorAll(\'a[href*="?t="]\').length')

async def _scroll_to_bottom(page: Page, pane_selector: Optional[str]) -> None:
    if pane_selector:
        pane = page.locator(pane_selector).first
        await pane.evaluate('(el) => el.scrollTo(0, el.scrollHeight)')
    else:
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')

async def _auto_hydrate_transcript(page: Page) -> None:
    """
    Wait for transcript anchors to appear, then scroll the transcript pane (or window)
    until the number of anchors stops increasing for HYDRATE_STABLE_ROUNDS checks.
    """
    # Wait for the transcript tab to render something anchor-like
    try:
        await page.wait_for_selector('a[href*="?t="]', timeout=10000)
    except Exception:
        # No anchors at all → proceed; parsing will fallback
        return

    # Choose the first pane selector that exists; else None (scroll window)
    pane_selector: Optional[str] = None
    for sel in TRANSCRIPT_PANE_SELECTORS:
        sel = sel.strip()
        if not sel:
            continue
        try:
            if await page.locator(sel).count() > 0:
                pane_selector = sel
                break
        except Exception:
            continue

    stable_rounds = 0
    last_count = await _count_transcript_anchors(page)

    for _ in range(AUTO_SCROLL_MAX_STEPS):
        await _scroll_to_bottom(page, pane_selector)
        await page.wait_for_timeout(AUTO_SCROLL_SLEEP_MS)
        cur_count = await _count_transcript_anchors(page)
        if cur_count == last_count:
            stable_rounds += 1
            if stable_rounds >= HYDRATE_STABLE_ROUNDS:
                break
        else:
            stable_rounds = 0
            last_count = cur_count

async def fetch_transcript_html(url: str) -> str:
    """
    Open the Komodo recording URL on the transcript tab, hydrate all rows by scrolling,
    then return the page HTML.
    """
    joiner = "&" if "?" in url else "?"
    transcript_url = url + f"{joiner}tab=transcript"
    async with context_page() as (_ctx, page):
        await safe_goto(page, transcript_url)
        await _auto_hydrate_transcript(page)
        html = await page.content()
        return html

# =========================
# Parsing (anchor-first, robust)
# =========================
# Timestamp regex: hh?:mm:ss or m:ss
_TS_RE = re.compile(r"^(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)$")

def _parse_hhmmss_to_seconds(ts: str) -> Optional[float]:
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        ts = f"0:{ts}"  # normalize m:ss → 0:m:ss
    m = _TS_RE.match(ts)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    return float(hh * 3600 + mm * 60 + ss)

def extract_segments(html: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Extract (title, segments[{start, text}]) using anchors with ?t= as row heads.
    We walk to the nearest grid-like container for text, remove the timestamp token,
    and collapse whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    segments: List[Dict[str, Any]] = []

    anchors = soup.select('a[href*="?t="]')
    for a in anchors:
        ts_text = (a.get_text(strip=True) or "").strip()
        # container row: nearest ancestor with grid-ish classes, else parent
        row = a.find_parent("div", class_=lambda c: c and "grid" in c) or a.parent
        text_block = " ".join(row.stripped_strings) if row else ts_text
        # Remove the first occurrence of timestamp token from the beginning
        if ts_text and text_block.startswith(ts_text):
            text_block = text_block[len(ts_text):].strip()
        # Sometimes anchor is in the middle; remove first occurrence anywhere
        elif ts_text and ts_text in text_block:
            text_block = text_block.replace(ts_text, "", 1).strip()
        text_block = re.sub(r"\s+", " ", text_block)
        if not text_block:
            continue

        start: Optional[float] = None
        if ts_text:
            start = _parse_hhmmss_to_seconds(ts_text)
        segments.append({"start": start, "text": text_block})

    # Fallback: if anchors empty, use generic text (last resort)
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
            segments[j]["text"] = segments[j]["text"][:max_chars]
            j += 1
        bounds.append((i, j))
        i = j
    return bounds

def join_text(segments: List[Dict[str, Any]], start: int, end: int) -> str:
    return "\n".join(seg["text"] for seg in segments[start:end])

# =========================
# Minimal ops endpoints (GET only; no HEAD)
# =========================
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": "Komodo Transcript Fetcher (Chunked)",
        "status": "ok",
        "health": "/health",
        "actions": {"fetchMeta": "POST /api/fetch-meta", "fetchChunk": "POST /api/fetch-chunk"}
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
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)
