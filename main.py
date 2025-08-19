import os
import re
import time
import json
import gzip
import asyncio
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# =========================
# Config (env overridable)
# =========================
GOTO_TIMEOUT_MS      = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))
ACTION_TIMEOUT_MS    = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))
FAST_SCROLL_STEPS    = int(os.getenv("FAST_SCROLL_STEPS", "4"))     # lighter
FULL_SCROLL_STEPS    = int(os.getenv("FULL_SCROLL_STEPS", "24"))    # lighter
SCROLL_PAUSE_MS      = int(os.getenv("SCROLL_PAUSE_MS", "200"))
MAX_CONCURRENCY      = int(os.getenv("MAX_CONCURRENCY", "1"))       # limit parallel scrapes

CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", "1200"))  # 20 minutes
CACHE_DIR            = Path(os.getenv("CACHE_DIR", "/tmp/komodo_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
SEM = asyncio.Semaphore(MAX_CONCURRENCY)

# =========================
# Health checks
# =========================
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
# Disk-backed cache helpers
# =========================
# In-memory index: url -> {"title": str, "path": Path, "ts": float}
INDEX: Dict[str, Dict[str, Any]] = {}
INDEX_LOCK = asyncio.Lock()

def _key_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def _file_for(url: str) -> Path:
    return CACHE_DIR / f"{_key_for(url)}.json.gz"

async def cache_get(url: str) -> Optional[Tuple[str, List[dict]]]:
    # Check index freshness; if fresh, load from disk
    async with INDEX_LOCK:
        meta = INDEX.get(url)
        if not meta:
            # If index lost (restart), but file exists, recover meta
            f = _file_for(url)
            if f.exists():
                # Use mtime as ts
                INDEX[url] = {"title": "(unknown)", "path": f, "ts": f.stat().st_mtime}
                meta = INDEX[url]
            else:
                return None
        if time.time() - meta["ts"] > CACHE_TTL_SECONDS:
            # Expired: delete file and index
            try:
                if Path(meta["path"]).exists():
                    Path(meta["path"]).unlink(missing_ok=True)
            finally:
                INDEX.pop(url, None)
            return None
        path = Path(meta["path"])
    if not path.exists():
        return None

    # Load gzipped JSON from disk
    def _load():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload["title"], payload["segments"]

    title, segments = await asyncio.to_thread(_load)
    return title, segments

async def cache_put(url: str, title: str, segments: List[dict]) -> None:
    path = _file_for(url)
    payload = {"title": title, "segments": segments, "ts": time.time()}

    def _save():
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    await asyncio.to_thread(_save)
    async with INDEX_LOCK:
        INDEX[url] = {"title": title, "path": path, "ts": time.time()}

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
    # mode: "fast" or "full"
    await click_transcript_tab(page)
    await autoscroll(page, steps=(FAST_SCROLL_STEPS if mode == "fast" else FULL_SCROLL_STEPS))

    # Try likely containers first
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

    # Fallback: lighter HTML text strip
    html = await page.content()
    txt = re.sub("<[^<]+?>", " ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
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
# Scrape (FAST/FULL) with disk cache
# =========================
async def scrape_title_and_segments(url: str, mode: str) -> Tuple[str, List[dict]]:
    # Try disk cache first
    cached = await cache_get(url)
    if cached:
        return cached[0], cached[1]

    async with SEM:  # limit concurrent Playwright sessions
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",  # use disk instead of /dev/shm
                    "--disable-gpu",
                    "--no-zygote",
                ]
            )
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
        # Use "fast" to be lighter; still returns correct chunk counts after normalization
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
@app.post("/api/fetch-chunk/")
async def fetch_chunk(req: ChunkReq):
    # If cached from meta, use cache; else scrape "full" once (heavier) then cache to disk
    cached = await cache_get(str(req.url))
    if cached:
        title, segs = cached
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
        "cached": True
    }
