import os
import re
import asyncio
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# -----------------------
# Tunables (env overrides)
# -----------------------
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))        # navigation timeout (default 60s)
ACTION_TIMEOUT_MS = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))    # actions/locator timeout (default 60s)
SCROLL_STEPS = int(os.getenv("SCROLL_STEPS", "40"))                 # how many times to scroll
SCROLL_PAUSE_MS = int(os.getenv("SCROLL_PAUSE_MS", "250"))          # pause between scrolls (ms)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Komodo Transcript Fetcher (Chunked)")

@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------
# Request models
# -----------------------
class MetaReq(BaseModel):
    url: HttpUrl
    max_chars: Optional[int] = 4500
    max_segments: Optional[int] = 250

class ChunkReq(BaseModel):
    url: HttpUrl
    chunk_index: int
    max_chars: Optional[int] = 4500
    max_segments: Optional[int] = 250

# -----------------------
# Playwright helpers
# -----------------------
TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))',
]

async def harden_page(page):
    """Speed up + make scraping resilient: block heavy resources, set timeouts, big viewport."""
    async def route_block(route):
        r = route.request
        if r.resource_type in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()
    await page.route("**/*", route_block)
    await page.set_viewport_size({"width": 1280, "height": 2000})
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.set_default_navigation_timeout(GOTO_TIMEOUT_MS)

async def goto_with_retry(page, url: str):
    """Navigate, then try with ?tab=transcript, trying different load states."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
    except PWTimeout:
        pass
    if "tab=transcript" not in url:
        joiner = "&" if "?" in url else "?"
        alt = f"{url}{joiner}tab=transcript"
        try:
            await page.goto(alt, wait_until="load", timeout=GOTO_TIMEOUT_MS)
        except PWTimeout:
            await page.goto(alt, wait_until="networkidle", timeout=GOTO_TIMEOUT_MS)

async def click_transcript_tab(page):
    for locator in ["text=Transcript", "role=tab[name='Transcript']", "button:has-text('Transcript')"]:
        try:
            if await page.locator(locator).count():
                await page.click(locator, timeout=ACTION_TIMEOUT_MS)
                break
        except Exception:
            pass

async def autoscroll(page, steps: int = SCROLL_STEPS, pause_ms: int = SCROLL_PAUSE_MS):
    for _ in range(steps):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(pause_ms / 1000.0)

async def extract_transcript(page) -> str:
    """Try likely transcript containers; fallback to body text; ultimate fallback to stripped HTML."""
    await click_transcript_tab(page)
    # trigger lazy-load by scrolling
    await autoscroll(page)

    # Primary: scoped containers
    for sel in TRANSCRIPT_SELECTORS:
        try:
            if await page.locator(sel).count():
                el = page.locator(sel).first
                await el.wait_for(timeout=ACTION_TIMEOUT_MS)
                text = await el.inner_text(timeout=ACTION_TIMEOUT_MS)
                if text and len(text.strip()) > 10:
                    return text
        except Exception:
            pass

    # Secondary: body text, after another scroll
    try:
        await autoscroll(page)
        return await page.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
    except Exception:
        # Ultimate: raw HTML -> strip tags
        html = await page.content()
        # Very crude tag strip to avoid extra deps
        text = re.sub("<[^<]+?>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

# -----------------------
# Normalization & chunking
# -----------------------
def normalize_lines_to_segments(text: str) -> List[dict]:
    """
    Convert raw transcript text into segments with optional timestamps.
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
            try:
                mm, ss = ts.split(":")
                sec = int(mm) * 60 + float(ss)
            except Exception:
                sec = None
            segments.append({"start": sec, "text": (rest or "").strip()})
        else:
            segments.append({"text": ln})
    return segments

def estimate_total_chars(segments: List[dict]) -> int:
    return sum(len(s.get("text", "")) for s in segments)

def chunk_ranges_by_limits(segments: List[dict], max_chars: int, max_segments: int) -> List[Tuple[int, int]]:
    """Return list of (start_index, end_index_inclusive) ranges that fit limits."""
    ranges = []
    i = 0
    n = len(segments)
    while i < n:
        chars = 0
        count = 0
        j = i
        while j < n and count < max_segments and (chars + len(segments[j].get("text", ""))) <= max_chars:
            chars += len(segments[j].get("text", ""))
            count += 1
            j += 1
        if j == i:  # single segment too large; force include 1 to make progress
            j = i + 1
        ranges.append((i, j - 1))
        i = j
    return ranges

def build_chunk_text(segments: List[dict], start_idx: int, end_idx: int) -> str:
    lines = []
    for k in range(start_idx, end_idx + 1):
        s = segments[k]
        t = s.get("text", "")
        if not t:
            continue
        # format timestamp if available
        ts = s.get("start", None)
        if ts is not None:
            mm = int(ts // 60)
            ss = int(ts % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {t}")
        else:
            lines.append(t)
    return "\n".join(lines)

# -----------------------
# Page load + segment extraction
# -----------------------
async def load_title_and_segments(url: str) -> Tuple[str, List[dict]]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        try:
            await harden_page(page)
            await goto_with_retry(page, url)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=GOTO_TIMEOUT_MS)
            except PWTimeout:
                pass
            text = await extract_transcript(page)
            title = await page.title()
        finally:
            await browser.close()

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")
    return title, normalize_lines_to_segments(text)

# -----------------------
# API routes
# -----------------------
@app.post("/api/fetch-meta")
async def fetch_meta(req: MetaReq):
    try:
        title, segs = await load_title_and_segments(str(req.url))
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")
    total = len(segs)
    est_chars = estimate_total_chars(segs)
    ranges = chunk_ranges_by_limits(segs, req.max_chars or 4500, req.max_segments or 250)
    return {
        "title": title,
        "total_segments": total,
        "estimated_total_chars": est_chars,
        "chunks_count": len(ranges),
        "max_chars": req.max_chars or 4500,
        "max_segments": req.max_segments or 250,
    }

@app.post("/api/fetch-chunk")
async def fetch_chunk(req: ChunkReq):
    try:
        title, segs = await load_title_and_segments(str(req.url))
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")

    ranges = chunk_ranges_by_limits(segs, req.max_chars or 4500, req.max_segments or 250)
    m = len(ranges)
    if m == 0:
        raise HTTPException(status_code=404, detail="No transcript content available to chunk.")
    if req.chunk_index < 0 or req.chunk_index >= m:
        raise HTTPException(status_code=416, detail=f"chunk_index out of range (0..{m-1})")

    start_idx, end_idx = ranges[req.chunk_index]
    text = build_chunk_text(segs, start_idx, end_idx)
    return {
        "title": title,
        "chunk_index": req.chunk_index,
        "chunks_count": m,
        "segments_in_chunk": end_idx - start_idx + 1,
        "segments_range": [start_idx, end_idx],
        "text": text,
        "segments": segs[start_idx:end_idx + 1],
    }
