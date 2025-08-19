from fastapi import FastAPI
from pydantic import BaseModel
from playwright.async_api import async_playwright
import uvicorn
import math
import hashlib
import asyncio

app = FastAPI()

# ==== MODELS ====

class FetchMetaRequest(BaseModel):
    url: str
    max_chars: int = 4500
    max_segments: int = 250

class FetchChunkRequest(BaseModel):
    url: str
    chunk_index: int
    max_chars: int = 4500
    max_segments: int = 250

# ==== SIMPLE IN-MEMORY CACHE ====
# Key = sha256(url), Value = {"title": str, "body_text": str}
CACHE = {}
CACHE_LOCK = asyncio.Lock()


# ==== UTILITIES ====

async def scrape_transcript(url: str):
    """Scrape title + transcript text from a Komodo link using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url, timeout=60000)

        # Grab page title
        title = await page.title()

        # Grab text content (replace selector with transcript container if needed)
        body_text = await page.inner_text("body")

        await context.close()
        await browser.close()

    return title, body_text


async def get_or_cache_transcript(url: str):
    """Return cached transcript if exists, otherwise scrape and store it."""
    key = hashlib.sha256(url.encode()).hexdigest()
    async with CACHE_LOCK:
        if key in CACHE:
            return CACHE[key]["title"], CACHE[key]["body_text"]

    # Scrape fresh if not cached
    title, body_text = await scrape_transcript(url)

    async with CACHE_LOCK:
        CACHE[key] = {"title": title, "body_text": body_text}

    return title, body_text


# ==== ROUTES ====

@app.get("/")
async def root():
    return {"message": "✅ Komodo Transcript API is running"}


@app.post("/api/fetch-meta")
async def fetch_meta(req: FetchMetaRequest):
    title, body_text = await get_or_cache_transcript(req.url)

    total_chars = len(body_text)
    total_segments = len(body_text.split())
    chunks_count = math.ceil(total_chars / req.max_chars)

    return {
        "title": title,
        "total_segments": total_segments,
        "estimated_total_chars": total_chars,
        "chunks_count": chunks_count,
        "max_chars": req.max_chars,
        "max_segments": req.max_segments,
    }


@app.post("/api/fetch-chunk")
async def fetch_chunk(req: FetchChunkRequest):
    title, body_text = await get_or_cache_transcript(req.url)

    total_chars = len(body_text)
    chunks_count = math.ceil(total_chars / req.max_chars)

    if req.chunk_index < 0 or req.chunk_index >= chunks_count:
        return {"error": "Invalid chunk index"}

    start = req.chunk_index * req.max_chars
    end = start + req.max_chars
    chunk_text = body_text[start:end]

    return {
        "title": title,
        "chunk_index": req.chunk_index,
        "chunk_text": chunk_text,
        "total_chunks": chunks_count,
        "max_chars": req.max_chars,
    }


# ==== MAIN ====

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
