import os
import re
import asyncio
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# --------------------
# Config via env vars
# --------------------
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))        # navigation timeout
ACTION_TIMEOUT_MS = int(os.getenv("ACTION_TIMEOUT_MS", "60000"))    # click/locator/inner_text timeout
SCROLL_STEPS = int(os.getenv("SCROLL_STEPS", "10"))                 # reduced default autoscroll loops
SCROLL_PAUSE_MS = int(os.getenv("SCROLL_PAUSE_MS", "250"))          # pause between scrolls (ms)

app = FastAPI()

# --------------------
# Health Checks
# --------------------
@app.get("/")
async def health():
    return {"status": "ok", "service": "komodo-transcript", "endpoints": ["/api/fetch-meta", "/api/fetch-chunk"]}

@app.head("/")
async def health_head():
    return Response(status_code=200)

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

async def safe_goto(page, url: str, retries: int = 3):
    """Navigate with retries and different wait conditions."""
    last_err = None
    for attempt in range(1, retries + 1):
        for cond in ["domcontentloaded", "load", "networkidle"]:
            try:
                await page.goto(url, wait_until=cond, timeout=GOTO_TIMEOUT_MS)
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

async def autoscroll(page, steps: int = SCROLL_STEPS, pause_ms: int = SCROLL_PAUSE_MS):
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

async def extract_transcript(page) -> str:
    """Try to extract transcript text with fallbacks."""
    await click_transcript_tab(page)
    await autoscroll(page)

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

    # Fallback: body text
    try:
        await autoscroll(page)
        return await page.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
    except Exception:
        html = await page.content()
        txt = re.sub("<[^<]+?>", " ", html)
        return re.sub(r"\s+", " ", txt).strip()

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

async def load_title_and_segments(url: str) -> Tuple[str, List[dict]]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        async with browser.new_context() as ctx:
            page = await ctx.new_page()
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
                text = await extract_transcript(page)
                title = await page.title()
            finally:
                await ctx.close()

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")
    return title, normalize_lines_to_segments(text)

# --------------------
# Endpoints
# --------------------
@app.post("/api/fetch-meta")
async def fetch_meta(req: MetaReq):
    try:
        title, segs = await load_title_and_segments(str(req.url))
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")

    total = len(segs)
    est_chars = sum(len(s.get("text", "")) for s in segs)

    max_chars = req.max_chars or 4500
    max_segments = req.max_segments or 250

    ranges = []
    i, n = 0, total
    while i < n:
        chars = 0
        count = 0
        j = i
        while j < n and count < max_segments and (chars + len(segs[j].get("text", ""))) <= max_chars:
            chars += len(segs[j].get("text", ""))
            count += 1
            j += 1
        if j == i:
            j = i + 1
        ranges.append((i, j - 1))
        i = j

    return {
        "title": title,
        "total_segments": total,
        "estimated_total_chars": est_chars,
        "chunks_count": len(ranges),
        "max_chars": max_chars,
        "max_segments": max_segments,
    }

@app.post("/api/fetch-chunk")
async def fetch_chunk(req: ChunkReq):
    try:
        title, segs = await load_title_and_segments(str(req.url))
    except PWTimeout:
        raise HTTPException(status_code=504, detail="Timeout loading transcript (page too slow or too large)")

    max_chars = req.max_chars or 4500
    max_segments = req.max_segments or 250

    ranges = []
    i, n = 0, len(segs)
    while i < n:
        chars = 0
        count = 0
        j = i
        while j < n and count < max_segments and (chars + len(segs[j].get("text", ""))) <= max_chars:
            chars += len(segs[j].get("text", ""))
            count += 1
            j += 1
        if j == i:
            j = i + 1
        ranges.append((i, j - 1))
        i = j

    m = len(ranges)
    if m == 0:
        raise HTTPException(status_code=404, detail="No transcript content available to chunk.")
    if req.chunk_index < 0 or req.chunk_index >= m:
        raise HTTPException(status_code=416, detail=f"chunk_index out of range (0..{m-1})")

    start_idx, end_idx = ranges[req.chunk_index]

    lines = []
    for k in range(start_idx, end_idx + 1):
        s = segs[k]
        t = s.get("text", "")
        if not t:
            continue
        ts = s.get("start")
        if ts is not None:
            mm = int(ts // 60)
            ss = int(ts % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {t}")
        else:
            lines.append(t)

    return {
        "title": title,
        "chunk_index": req.chunk_index,
        "chunks_count": m,
        "segments_in_chunk": end_idx - start_idx + 1,
        "segments_range": [start_idx, end_idx],
        "text": "\n".join(lines),
        "segments": segs[start_idx:end_idx + 1],
    }
