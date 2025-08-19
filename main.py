import math, time
from collections import OrderedDict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright

app = FastAPI()

# Globals
browser = None
playwright = None

# Cache: url -> { "title": str, "segments": list[str], "last_used": float }
# OrderedDict to preserve usage order
MAX_CACHE_ITEMS = 3
CACHE_TTL = 1800  # 30 minutes
cache = OrderedDict()

def touch_cache(url: str):
    """Update last_used and move to end (most recent)."""
    if url in cache:
        cache[url]["last_used"] = time.time()
        cache.move_to_end(url)

def evict_if_needed():
    """Drop expired or excess cache entries."""
    now = time.time()
    # Drop expired
    expired = [u for u,v in cache.items() if now - v["last_used"] > CACHE_TTL]
    for u in expired:
        print(f"🗑️ Expired cache entry: {u}")
        cache.pop(u, None)

    # Drop oldest if over limit
    while len(cache) > MAX_CACHE_ITEMS:
        old_url, _ = cache.popitem(last=False)
        print(f"🗑️ Evicted LRU entry: {old_url}")

# Startup: one browser
@app.on_event("startup")
async def startup_event():
    global browser, playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    print("✅ Browser launched")

@app.on_event("shutdown")
async def shutdown_event():
    global browser, playwright
    if browser:
        await browser.close()
    if playwright:
        await playwright.stop()
    print("🛑 Browser closed")

# Request models
class FetchMetaRequest(BaseModel):
    url: str
    max_chars: int = 4500
    max_segments: int = 250

class FetchChunkRequest(BaseModel):
    url: str
    chunk_index: int
    max_chars: int = 4500
    max_segments: int = 250

# Helpers
async def extract_segments(url: str):
    global browser
    page = await browser.new_page()
    await page.goto(url, timeout=60000)
    segments = await page.eval_on_selector_all(
        "div.transcript-segment", "els => els.map(e => e.innerText)"
    )
    title = await page.title()
    await page.close()
    return title, segments

# API
@app.post("/api/fetch-meta")
async def fetch_meta(req: FetchMetaRequest):
    try:
        evict_if_needed()

        if req.url in cache:
            print("♻️ Cache hit for", req.url)
            data = cache[req.url]
            touch_cache(req.url)
        else:
            print("🔎 Scraping", req.url)
            title, segments = await extract_segments(req.url)
            data = {"title": title, "segments": segments, "last_used": time.time()}
            cache[req.url] = data
            touch_cache(req.url)
            evict_if_needed()

        total_segments = len(data["segments"])
        estimated_total_chars = sum(len(s) for s in data["segments"])
        chunks_count = math.ceil(total_segments / req.max_segments)

        return JSONResponse(content={
            "title": data["title"],
            "total_segments": total_segments,
            "estimated_total_chars": estimated_total_chars,
            "chunks_count": chunks_count,
            "max_chars": req.max_chars,
            "max_segments": req.max_segments
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/fetch-chunk")
async def fetch_chunk(req: FetchChunkRequest):
    try:
        evict_if_needed()

        if req.url not in cache:
            print("⚠️ Not cached, scraping first:", req.url)
            title, segments = await extract_segments(req.url)
            cache[req.url] = {"title": title, "segments": segments, "last_used": time.time()}
            touch_cache(req.url)
            evict_if_needed()

        data = cache[req.url]
        touch_cache(req.url)

        start = req.chunk_index * req.max_segments
        end = start + req.max_segments
        chunk_segments = data["segments"][start:end]
        chars_count = sum(len(s) for s in chunk_segments)

        return JSONResponse(content={
            "chunk_index": req.chunk_index,
            "segments": chunk_segments,
            "chars_count": chars_count
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
