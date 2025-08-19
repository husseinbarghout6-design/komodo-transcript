import os
import re
import time
import asyncio
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# =========================
# Config (env overridable)
# =========================
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))       # navigation timeout
ACTION_TIMEOUT_MS = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))   # element ops timeout

FAST_SCROLL_STEPS = int(os.getenv("FAST_SCROLL_STEPS", "8"))       # fast mode (meta)
FULL_SCROLL_STEPS = int(os.getenv("FULL_SCROLL_STEPS", "60"))      # full mode (chunk)
SCROLL_PAUSE_MS   = int(os.getenv("SCROLL_PAUSE_MS", "200"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ITEMS   = int(os.getenv("CACHE_MAX_ITEMS", "3"))

UI_NOISE_PREFIXES = [
    "Record in Browser", "Komodo Blog", "Pricing", "Login", "Get Komodo Free",
    "This is a modal window.", "No compatible source was found for this media.",
    "Highlights", "Transcript", "Feedback", "Copy", "Chapters", "Annotate"
]

TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))'
]

app = FastAPI()

# Health checks
@app.get("/")
async def health():
    return {"status": "ok", "service": "komodo-transcript", "endpoints": ["/api/fetch-meta", "/api/fetch-chunk"]}

@app.head("/")
async def health_head():
    return Response(status_code=200)

# =========================
# Schemas
# =========================
class MetaReq(BaseModel):
    url: HttpUrl
    max_chars: int = 4500
    max_segments: int = 250

class ChunkReq(MetaReq):
    chunk_index: int

# =========================
# Small in-memory cache
# =========================
# cache[url] = {"title": str, "segments": list[dict], "ts": epoch}
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = asyncio.Lock()

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

async def cache_put(url: str, title: str, segments: List[dict]):
    async with CACHE_LOCK:
        # evict oldest if over capacity
        if len(CACHE) >= CACHE_MAX_ITEMS:
            oldest_key = min(CACHE.keys(), key=lambda k: CACHE[k]["ts"])
            try: del CACHE[oldest_key]
            except Exception: pass
        CACHE[url] = {"title": title, "segments": segments, "ts": time.time()}

# =========================
# Playwright helpers
# =========================
async def harden_page(page):
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
    for locator in ["text=Transcript", "role=tab[name='Transcript']", "button:has-text('Transcript')"]:
        try:
            if await page.locator(locator).count():
                await page.click(locator, timeout=ACTION_TIMEOUT_MS)
                break
        except Exception:
            pass

async def autoscroll(page, steps: int, pause_ms: int = SCROLL_PAUSE_MS):
    for _ in range(steps):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(pause_ms / 1000.0)

async def extract_transcript(page, mode: str) -> str:
    """
    Extract transcript text.
    mode="fast": minimal scroll + HTML strip fallback only
    mode="full": deeper scroll + body.inner_text fallback
    """
    await click_transcript_tab(page)
    await autoscroll(page, steps=(FAST_SCROLL_STEPS if mode == "fast" else FULL_SCROLL_STEPS))

    # Likely containers first
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

    # Fallbacks
    html = await page.content()
    txt = re.sub("<[^<]+?>", " ", html)
    txt = re.sub(r"\s+", " ", txt).strip()

    if mode == "fast":
        # cap size to avoid heavy strings in meta
        return txt[:200000]
    else:
        # try a heavier body.inner_text only in full mode
        try:
            await autoscroll(page, steps=max(10, FULL_SCROLL_STEPS // 2))
            body_text = await page.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
            if body_text and len(body_text.strip()) > 10:
                return body_text
        except Exception:
            pass
        return txt

# =========================
# Parsing & chunking
# =========================
def is_noise_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    for p in UI_NOISE_PREFIXES:
        if s == p or s.startswith(p):
            return True
    return False

def split_heavy_line(line: str) -> List[str]:
    # Split by Arabic/English punctuation (., !, ?, ؟, ؛, ،)
    parts = re.split(r'(?<=[\.\!\?؟؛،])\s+', line)
    out = [p.strip() for p in parts if p and p.strip()]
    return out if out else [line]

def normalize_lines_to_segments(text: str) -> List[dict]:
    segments: List[dict] = []
    raw_lines = re.split(r'\r+|\n+|\s{4,}', text)

    for raw in raw_lines:
        ln = (raw or "").strip()
        if not ln or is_noise_line(ln):
            continue

        # [mm:ss] prefix (optional decimal seconds)
        m = re.match(r'^\[?(\d{1,2}:\d{2}(?:\.\d{1,2})?)\]?\s*(.*)$', ln)
        if m:
            ts, rest = m.groups()
            mm_ss = (rest or "").strip()
            try:
                mm, ss = ts.split(":")
                sec = int(mm) * 60 + float(ss)
            except Exception:
                sec = None
            if not mm_ss:
                continue
            if len(mm_ss) > 500:
                for piece in split_heavy_line(mm_ss):
                    segments.append({"start": sec, "text": piece})
            else:
                segments.append({"start": sec, "text": mm_ss})
            continue

        if len(ln) > 500:
            for piece in split_heavy_line(ln):
                if piece and not is_noise_line(piece):
                    segments.append({"text": piece})
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

# =========================
# Scrape (FAST/FULL) + cache
# =========================
async def scrape_title_and_segments(url: str, mode: str) -> Tuple[str, List[dict]]:
    # cache?
    cached = cache_get(url)
    if cached:
        return cached["title"], cached["segments"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()
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
            await context.close()
            await browser.close()

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")

    segments = normalize_lines_to_segments(text)
    await cache_put(url, title, segments)
    return title, segments

# =========================
# Endpoints
# =========================
@app.post("/api/fetch-meta")
@app.post("/api/fetch-meta/")
async def fetch_meta(req: MetaReq):
    try:
        # Use FULL once so we actually see transcript in meta too
        title, segs = await scrape_title_and_segments(str(req.url), mode="full")
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
@app.post("/api/fetch-chunk/")
async def fetch_chunk(req: ChunkReq):
    # If cached from meta, use cache; else scrape FULL
    cached = cache_get(str(req.url))
    if cached:
        title, segs = cached["title"], cached["segments"]
    else:
        try:
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
