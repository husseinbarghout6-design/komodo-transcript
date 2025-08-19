import os
import re
import time
import math
import asyncio
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# =========================
# Config (env-tunable)
# =========================
PORT = int(os.getenv("PORT", "10000"))
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))        # navigation timeout
ACTION_TIMEOUT_MS = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))    # locator/click/inner_text timeout

# Light scroll for meta, deeper scroll on a single pass for chunk caching
FAST_SCROLL_STEPS = int(os.getenv("FAST_SCROLL_STEPS", "4"))
FULL_SCROLL_STEPS = int(os.getenv("FULL_SCROLL_STEPS", "14"))
SCROLL_PAUSE_MS = int(os.getenv("SCROLL_PAUSE_MS", "200"))

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))            # 1 is safest on 512MB
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "3"))            # LRU cache entries
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800"))                     # seconds (30 min)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Komodo Transcript (chunked)")

# Global Playwright handles and a concurrency guard
_playwright = None
_browser = None
SEM = asyncio.Semaphore(MAX_CONCURRENCY)

# Simple LRU cache: url -> {"title": str, "segments": List[dict], "last_used": float}
_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

# =========================
# Models
# =========================
class MetaReq(BaseModel):
    url: HttpUrl
    max_chars: int = 4500
    max_segments: int = 250

class ChunkReq(MetaReq):
    chunk_index: int

# =========================
# Startup / Shutdown
# =========================
@app.on_event("startup")
async def on_startup():
    global _playwright, _browser
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--no-zygote",
            "--mute-audio",
        ],
    )
    print("✅ Chromium launched; server ready.")

@app.on_event("shutdown")
async def on_shutdown():
    global _playwright, _browser
    try:
        if _browser:
            await _browser.close()
    finally:
        if _playwright:
            await _playwright.stop()
    print("🛑 Shutdown complete.")

# =========================
# Health / Root (primary)
# =========================
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "komodo-transcript",
        "endpoints": ["/", "/api/fetch-meta", "/api/fetch-chunk"],
    }

@app.head("/")
async def health_head():
    return Response(status_code=200)

@app.post("/")
async def root_post(req: MetaReq):
    """Alias: treat POST / as primary action for fetch-meta."""
    return await fetch_meta(req)

# =========================
# Helpers: page & scraping
# =========================
async def _new_context():
    """Create a new context with resource blocking to save memory."""
    ctx = await _browser.new_context(viewport={"width": 1280, "height": 2000})
    ctx.set_default_timeout(ACTION_TIMEOUT_MS)
    ctx.set_default_navigation_timeout(GOTO_TIMEOUT_MS)
    # Block heavy resources
    async def route_block(route):
        r = route.request
        if r.resource_type in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()
    await ctx.route("**/*", route_block)
    return ctx

async def safe_goto(page, url: str):
    """Navigate to URL with a few strategies to reduce flakiness."""
    last = None
    for wait_state in ("domcontentloaded", "load", "networkidle"):
        try:
            await page.goto(url, wait_until=wait_state, timeout=GOTO_TIMEOUT_MS)
            return
        except Exception as e:
            last = e
    raise last if last else RuntimeError("Navigation failed")

async def autoscroll(page, steps: int, pause_ms: int = SCROLL_PAUSE_MS):
    for _ in range(max(0, steps)):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(pause_ms / 1000.0)

TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))',
]

async def click_transcript_tab(page):
    for locator in ["text=Transcript", "role=tab[name='Transcript']", "button:has-text('Transcript')"]:
        try:
            if await page.locator(locator).count():
                await page.click(locator, timeout=ACTION_TIMEOUT_MS)
                await asyncio.sleep(0.2)
                break
        except Exception:
            pass

async def extract_transcript_text(url: str, deep_scroll: bool) -> Tuple[str, str]:
    """
    Return (title, full_text).
    deep_scroll=False is used for quick meta checks, True for a single heavy pass to cache.
    """
    async with SEM:
        ctx = await _new_context()
        try:
            page = await ctx.new_page()
            # Try transcript tab first
            await safe_goto(page, url)
            if "tab=transcript" not in url:
                joiner = "&" if "?" in url else "?"
                try:
                    await safe_goto(page, url + f"{joiner}tab=transcript")
                except Exception:
                    # Fine; fall back to current page
                    pass

            await click_transcript_tab(page)
            await autoscroll(page, FAST_SCROLL_STEPS if not deep_scroll else FULL_SCROLL_STEPS)

            # First, try likely transcript containers
            for sel in TRANSCRIPT_SELECTORS:
                try:
                    loc = page.locator(sel).first
                    if await loc.count():
                        await loc.wait_for(timeout=ACTION_TIMEOUT_MS)
                        txt = await loc.inner_text(timeout=ACTION_TIMEOUT_MS)
                        if txt and len(txt.strip()) > 10:
                            return (await page.title()), txt
                except Exception:
                    continue

            # Fallback: whole body (can be heavy)
            await autoscroll(page, 2 if not deep_scroll else 6)
            body_txt = await page.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
            return (await page.title()), body_txt
        finally:
            await ctx.close()

# =========================
# Helpers: normalization & chunking
# =========================
TIMESTAMP_RE = re.compile(r'^\[?(\d{1,2}:\d{2}(?:\.\d{1,2})?)\]?\s*(.*)$')

def _normalize_text_to_segments(text: str) -> List[Dict[str, Any]]:
    """
    Produce a list of {start?: seconds|None, text: str}.
    1) Prefer lines that start with a timestamp like 00:23 or [01:05.2].
    2) Otherwise use non-empty lines.
    3) If it still looks too sparse, split long lines on punctuation to create ~phrase segments.
    """
    segments: List[Dict[str, Any]] = []
    # Remove some obvious UI noise
    ui_noise = {
        "Record in Browser", "Komodo Blog", "Pricing", "Login", "Get Komodo Free",
        "Highlights", "Transcript", "Feedback", "Copy", "Annotate", "Chapters",
        "This is a modal window.", "No compatible source was found for this media.",
    }

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and ln not in ui_noise]

    # First pass: timestamps / plain lines
    for ln in lines:
        m = TIMESTAMP_RE.match(ln)
        if m:
            ts, rest = m.groups()
            start = None
            try:
                mm, ss = ts.split(":")
                start = int(mm) * 60 + float(ss)
            except Exception:
                start = None
            segments.append({"start": start, "text": (rest or "").strip()})
        else:
            segments.append({"text": ln})

    # If we accidentally made huge blobs, further split long items to ~200 chars
    refined: List[Dict[str, Any]] = []
    for s in segments:
        t = s.get("text", "")
        if len(t) > 220:
            # Split at punctuation or spaces
            parts = re.split(r'(?<=[\.\!\?،؛])\s+|(?<=,)\s+|(?<=;)\s+', t)
            if len(parts) == 1:  # still one big chunk; hard split
                for i in range(0, len(t), 200):
                    refined.append({"start": s.get("start"), "text": t[i:i+200].strip()})
            else:
                for p in parts:
                    p = p.strip()
                    if p:
                        refined.append({"start": s.get("start"), "text": p})
        else:
            refined.append(s)

    # Final clean: remove empty texts
    refined = [s for s in refined if s.get("text", "").strip()]
    return refined

def _build_chunk_ranges(segs: List[Dict[str, Any]], max_chars: int, max_segments: int) -> List[Tuple[int, int]]:
    """
    Greedy ranges that satisfy both constraints:
      - total characters per chunk <= max_chars
      - number of segments per chunk <= max_segments
    Always make progress (at least 1 segment).
    """
    ranges: List[Tuple[int, int]] = []
    n = len(segs)
    i = 0
    while i < n:
        chars = 0
        count = 0
        j = i
        while j < n and count < max_segments:
            next_len = len(segs[j].get("text", ""))
            if count > 0 and chars + next_len > max_chars:
                break
            chars += next_len
            count += 1
            j += 1
        if j == i:  # extremely long single segment; force move by 1
            j = i + 1
        ranges.append((i, j - 1))
        i = j
    return ranges

def _format_mmss(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"{mm:02d}:{ss:02d}"
    except Exception:
        return None

# =========================
# Cache helpers
# =========================
def _touch_cache(url: str):
    if url in _cache:
        _cache[url]["last_used"] = time.time()
        _cache.move_to_end(url)

def _evict_if_needed():
    # Drop expired
    now = time.time()
    expired = [u for u, v in _cache.items() if now - v["last_used"] > CACHE_TTL]
    for u in expired:
        _cache.pop(u, None)
        print(f"🗑️ cache expired: {u}")

    # LRU over capacity
    while len(_cache) > MAX_CACHE_ITEMS:
        old_url, _ = _cache.popitem(last=False)
        print(f"🗑️ cache evicted LRU: {old_url}")

async def _get_or_scrape(url: str, deep_scroll: bool) -> Tuple[str, List[Dict[str, Any]]]:
    """
    If in cache, return it. Otherwise scrape once, normalize, store, and return.
    deep_scroll=True will scroll more to load larger transcripts (one time).
    """
    _evict_if_needed()
    if url in _cache:
        _touch_cache(url)
        data = _cache[url]
        return data["title"], data["segments"]

    title, full_txt = await extract_transcript_text(url, deep_scroll=deep_scroll)
    if not full_txt or len(full_txt.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is it public?)")

    segs = _normalize_text_to_segments(full_txt)
    if not segs:
        raise HTTPException(status_code=404, detail="Transcript found but no usable lines.")

    _cache[url] = {"title": title, "segments": segs, "last_used": time.time()}
    _touch_cache(url)
    _evict_if_needed()
    return title, segs

# =========================
# API: fetch-meta & fetch-chunk
# =========================
@app.post("/api/fetch-meta")
async def fetch_meta(req: MetaReq):
    """
    Light operation: quick scroll, read transcript, normalize -> compute chunk ranges.
    """
    try:
        title, segs = await _get_or_scrape(str(req.url), deep_scroll=False)
        total_segments = len(segs)
        estimated_total_chars = sum(len(s.get("text", "")) for s in segs)

        ranges = _build_chunk_ranges(segs, max_chars=max(1, req.max_chars), max_segments=max(1, req.max_segments))
        return {
            "title": title,
            "total_segments": total_segments,
            "estimated_total_chars": estimated_total_chars,
            "chunks_count": len(ranges),
            "max_chars": req.max_chars,
            "max_segments": req.max_segments,
        }
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch-meta error: {e!r}")

@app.post("/api/fetch-chunk")
async def fetch_chunk(req: ChunkReq):
    """
    Heavier once per URL: do a deeper scroll if not cached yet, then serve chunks from memory.
    """
    try:
        title, segs = await _get_or_scrape(str(req.url), deep_scroll=True)

        ranges = _build_chunk_ranges(segs, max_chars=max(1, req.max_chars), max_segments=max(1, req.max_segments))
        m = len(ranges)
        if m == 0:
            raise HTTPException(status_code=404, detail="No transcript content available to chunk.")
        if req.chunk_index < 0 or req.chunk_index >= m:
            raise HTTPException(status_code=416, detail=f"chunk_index out of range (0..{m-1})")

        start_idx, end_idx = ranges[req.chunk_index]
        slice_segs = segs[start_idx:end_idx + 1]

        # Build plain text with mm:ss if present
        lines: List[str] = []
        for s in slice_segs:
            t = s.get("text", "").strip()
            if not t:
                continue
            ts = _format_mmss(s.get("start"))
            if ts:
                lines.append(f"[{ts}] {t}")
            else:
                lines.append(t)

        return {
            "title": title,
            "chunk_index": req.chunk_index,
            "chunks_count": m,
            "segments_in_chunk": end_idx - start_idx + 1,
            "segments_range": [start_idx, end_idx],
            "text": "\n".join(lines),
            "segments": slice_segs,
        }
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch-chunk error: {e!r}")
