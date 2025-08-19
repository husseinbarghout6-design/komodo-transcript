import os
import re
import asyncio
import time
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# --------------------
# Config via env vars
# --------------------
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))        # navigation timeout
ACTION_TIMEOUT_MS = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))    # click/locator/inner_text timeout

# FAST/ FULL scroll strategies
FAST_SCROLL_STEPS = int(os.getenv("FAST_SCROLL_STEPS", "8"))        # for /api/fetch-meta (keep small for speed)
FULL_SCROLL_STEPS = int(os.getenv("FULL_SCROLL_STEPS", "60"))       # for /api/fetch-chunk (more thorough)
SCROLL_PAUSE_MS = int(os.getenv("SCROLL_PAUSE_MS", "200"))

# Cache: keep last N pages for TTL seconds
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "3"))

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok", "service": "komodo-transcript", "endpoints": ["/api/fetch-meta", "/api/fetch-chunk"]}

# --------------------
# Request Schemas
# --------------------
class MetaReq(BaseModel):
    url: HttpUrl
    max_chars: int = 4500
    max_segments: int = 250

class ChunkReq(MetaReq):
    chunk_index: int

# --------------------
# Small in-memory cache (LRU-ish)
# --------------------
# cache[url] = {"title": str, "segments": list[dict], "ts": epoch}
CACHE: Dict[str, Dict[str, Any]] = {}

def cache_get(url: str) -> Optional[Dict[str, Any]]:
    item = CACHE.get(url)
    if not item:
        return None
    if time.time() - item["ts"] > CACHE_TTL_SECONDS:
        try:
            del CACHE[url]
        except Exception:
            pass
        return None
    return item

def cache_put(url: str, title: str, segments: List[dict]):
    # evict if too big
    if len(CACHE) >= CACHE_MAX_ITEMS:
        # remove oldest
        oldest_key = min(CACHE.keys(), key=lambda k: CACHE[k]["ts"])
        try:
            del CACHE[oldest_key]
        except Exception:
            pass
    CACHE[url] = {"title": title, "segments": segments, "ts": time.time()}

# --------------------
# Playwright Helpers
# --------------------
async def harden_page(page):
    """Block heavy assets, set timeouts & viewport."""
    async def route_block(route):
        r = route.request
        if r.resource_type in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()
    await page.route("**/*", route_block)
    await page.set_viewport_size({"width": 1280, "height": 2000})
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.set_default_navigation_timeout(GOTO_TIMEOUT_MS)

async def safe_goto(page, url: str, retries: int = 2):
    """Navigate with retries and different wait conditions."""
    last_err = None
    for attempt in range(1, retries + 1):
        for state in ("domcontentloaded", "load", "networkidle"):
            try:
                await page.goto(url, wait_until=state, timeout=GOTO_TIMEOUT_MS)
                return
            except Exception as e:
                last_err = e
        await asyncio.sleep(min(2 * attempt, 6))
    raise last_err if last_err else RuntimeError("Navigation failed")

async def click_transcript_tab(page):
    """Click the transcript tab if it exists."""
    for locator in ["text=Transcript", "role=tab[name='Transcript']", "button:has-text('Transcript')"]:
        try:
            if await page.locator(locator).count():
                await page.click(locator, timeout=ACTION_TIMEOUT_MS)
                break
        except Exception:
            pass

async def autoscroll(page, steps: int, pause_ms: int = SCROLL_PAUSE_MS):
    """Scroll to load lazy content."""
    for _ in range(steps):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(pause_ms / 1000.0)

TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))',
]

async def extract_transcript(page, mode: str) -> str:
    """
    Try to extract transcript text with fallbacks.
    mode = "fast"  -> minimal scrolling, no heavy body fallback
    mode = "full"  -> deeper scrolling + body fallback
    """
    await click_transcript_tab(page)
    steps = FAST_SCROLL_STEPS if mode == "fast" else FULL_SCROLL_STEPS
    await autoscroll(page, steps=steps)

    # Try likely containers
    for sel in TRANSCRIPT_SELECTORS:
        try:
            if await page.locator(sel).count():
                el = page.locator(sel).first
                await el.wait_for(timeout=ACTION_TIMEOUT_MS)
                txt = await el.inner_text(timeout=ACTION_TIMEOUT_MS)
                if txt and len(txt.strip()) > 10:
                    return txt
        except Exception:
            pass

    if mode == "full":
        # Fallback: body text (heavier)
        try:
            await autoscroll(page, steps=max(10, steps // 2))
            return await page.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
        except Exception:
            html = await page.content()
            txt = re.sub("<[^<]+?>", " ", html)
            return re.sub(r"\s+", " ", txt).strip()
    else:
        # FAST mode: avoid heavy body.inner_text; do a light HTML strip only
        html = await page.content()
        txt = re.sub("<[^<]+?>", " ", html)
        # Keep it brief in FAST mode
        return re.sub(r"\s+", " ", txt)[:200000].strip()  # cap to avoid huge strings

# --------------------
# Transcript Parsing
# --------------------
def normalize_lines_to_segments(text: str) -> List[dict]:
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

def chunk_ranges_by_limits(segments: List[dict], max_chars: int, max_segments: int):
    ranges = []
    i, n = 0, len(segments)
    while i < n:
        chars = 0
        count = 0
        j = i
        while j < n and count < max_segments and (chars + len(segments[j].get("text", ""))) <= max_chars:
            chars += len(segments[j].get("text", ""))
            count += 1
            j += 1
        if j == i:
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
        ts = s.get("start")
        if ts is not None:
            mm = int(ts // 60); ss = int(ts % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {t}")
        else:
            lines.append(t)
    return "\n".join(lines)

# --------------------
# Loaders (FAST / FULL) + Cache
# --------------------
async def scrape_title_and_segments(url: str, mode: str) -> Tuple[str, List[dict]]:
    """
    Scrape in 'fast' or 'full' mode. 'full' also writes to cache.
    """
    # Cache hit?
    if mode in ("fast", "full"):
        cached = cache_get(url)
        if cached:
            return cached["title"], cached["segments"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        try:
            await harden_page(page)
            await safe_goto(page, url)
            if "tab=transcript" not in url:
                joiner = "&" if "?" in url else "?"
                await safe_goto(page, url + f"{joiner}tab=transcript")

            try:
                await page.wait_for_load_state("domcontentloaded", timeout=GOTO_TIMEOUT_MS)
            except PWTimeout:
                pass

            text = await extract_transcript(page, mode=mode)
            title = await page.title()
        finally:
            await browser.close()

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")

    segments = normalize_lines_to_segments(text)

    # Only cache on 'full' to avoid partial fast-mode artifacts
    if mode == "full":
        cache_put(url, title, segments)

    return title, segments

# --------------------
# Endpoints
# --------------------
@app.post("/api/fetch-meta")
async def fetch_meta(req: MetaReq):
    """
    FAST mode: designed to return within ~10-20s for very large pages to satisfy GPT time budgets.
    """
    try:
        title, segs = await scrape_title_and_segments(str(req.url), mode="fast")
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")

    total = len(segs)
    est_chars = sum(len(s.get("text", "")) for s in segs)
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
    """
    FULL mode: thorough scrape and cached results for subsequent chunk calls.
    """
    # Try cache first for speed
    cached = cache_get(str(req.url))
    if cached:
        title, segs = cached["title"], cached["segments"]
    else:
        try:
            # Do a full scrape (heavier) ONCE, then cache
            title, segs = await scrape_title_and_segments(str(req.url), mode="full")
        except PWTimeout:
            raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")

    max_chars = req.max_chars or 4500
    max_segments = req.max_segments or 250

    ranges = chunk_ranges_by_limits(segs, max_chars, max_segments)
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
        "cached": cached is not None
    }
