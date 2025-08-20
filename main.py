import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Iterable, Dict

from fastapi import FastAPI, HTTPException, Query, APIRouter
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, HttpUrl
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
app = FastAPI(title="Komodo Transcript Service", version="1.1.0")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Render/Playwright tuning
# =========================
MAX_PW_CONTEXTS = int(os.getenv("MAX_PW_CONTEXTS", "2"))  # keep low on 512MB RAM
PW_NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "30000"))

_browser: Optional[Browser] = None
_pw = None
_pw_sema = asyncio.Semaphore(MAX_PW_CONTEXTS)

# Start one shared Chromium on startup; close it on shutdown
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
# Models
# =========================
class FetchMetaResponse(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_site_name: Optional[str] = None
    transcript_text_preview: Optional[str] = None
    raw_length: int
    notes: Optional[str] = None

class FetchMetaBody(BaseModel):
    url: HttpUrl

class ProcessRequest(BaseModel):
    chunks: List[str]
    max_chars: int = 4500
    max_segments: int = 250

class ProcessBatchLog(BaseModel):
    processed_of_total: str
    this_batch_segments: int
    this_batch_chars: int

class ProcessResponse(BaseModel):
    total_chunks: int
    batches: List[ProcessBatchLog]
    message: str

class FetchChunkBody(BaseModel):
    chunk: str
    max_chars: int = 4500
    max_segments: int = 250

# =========================
# Helpers
# =========================
async def safe_goto(page: Page, url: str, timeout_ms: int = PW_NAV_TIMEOUT_MS) -> None:
    """
    Navigate to a URL with reasonable waiting logic and better error surfacing.
    """
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeoutError as e:
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms // 2)
        except Exception as e2:
            raise HTTPException(status_code=504, detail=f"Timeout navigating to {url}: {e2}") from e

async def fetch_transcript_html(url: str) -> str:
    """
    Open the provided Komodo recording URL and switch to its transcript tab,
    then return the page HTML.
    """
    joiner = "&" if "?" in url else "?"
    transcript_url = url + f"{joiner}tab=transcript"

    async with context_page() as (_ctx, page):
        await safe_goto(page, transcript_url)
        # brief wait for dynamic content
        await page.wait_for_timeout(800)
        html = await page.content()
        return html

def extract_meta(html: str) -> Dict[str, Optional[str]]:
    """
    Extract useful metadata and a preview of transcript text from the HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    def meta_content(name: str = None, prop: str = None) -> Optional[str]:
        if name:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        if prop:
            tag = soup.find("meta", attrs={"property": prop})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    og_title = meta_content(prop="og:title")
    og_desc = meta_content(prop="og:description")
    og_site = meta_content(prop="og:site_name")
    desc = meta_content(name="description")

    body_text = soup.get_text(separator="\n", strip=True)
    preview = None
    if body_text:
        preview = body_text[:600] + ("..." if len(body_text) > 600 else "")

    return {
        "title": title,
        "description": desc,
        "og_title": og_title,
        "og_description": og_desc,
        "og_site_name": og_site,
        "transcript_preview": preview,
        "raw_len": len(body_text or ""),
    }

def pack_batches(
    chunks: List[str],
    max_chars: int = 4500,
    max_segments: int = 250,
) -> Iterable[Tuple[int, int, List[str]]]:
    """
    Yield (start_idx, end_idx_exclusive, batch) respecting both limits.
    Handles oversize single-chunk cases by truncating with a marker.
    """
    i = 0
    n = len(chunks)
    while i < n:
        used_chars = 0
        segs = 0
        j = i
        batch: List[str] = []
        while j < n:
            c = chunks[j]
            c_len = len(c)

            # If single chunk exceeds max_chars and batch empty, truncate it
            if c_len > max_chars and segs == 0:
                truncated = c[:max_chars]
                yield (j, j + 1, [truncated + "\n[TRUNCATED]"])
                j += 1
                i = j
                break

            if segs + 1 > max_segments or used_chars + c_len > max_chars:
                break

            batch.append(c)
            segs += 1
            used_chars += c_len
            j += 1

        if batch:
            yield (i, j, batch)
            i = j

        # Safety to force progress
        if not batch and j == i and i < n:
            oversized = chunks[i][:max_chars]
            yield (i, i + 1, [oversized + "\n[FORCED TRUNCATION]"])
            i += 1

# =========================
# Routes
# =========================
@app.get("/", include_in_schema=False)
async def index():
    return JSONResponse({
        "service": "Komodo Transcript Service",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "fetch_meta_get": "/fetch-meta?url=<public_komodo_recording_url>",
            "fetch_meta_post": "/fetch-meta",
            "fetch_chunk_post": "/fetch-chunk",
            "process_post": "/process",
        }
    })

@app.get("/home", include_in_schema=False)
async def home():
    return RedirectResponse(url="/docs", status_code=302)

@app.get("/health")
async def health():
    # also indicates browser status
    return {"status": "ok", "browser_initialized": _browser is not None}

# Unified router for GET/POST variants and API prefix mounting
router = APIRouter()

@router.get("/fetch-meta", response_model=FetchMetaResponse)
@router.get("/fetch-meta/", response_model=FetchMetaResponse)
async def fetch_meta_get(url: str = Query(..., description="Public Komodo recording URL")):
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Please provide a valid http(s) URL.")
    html = await fetch_transcript_html(url)
    meta = extract_meta(html)
    return FetchMetaResponse(
        url=url,
        title=meta.get("title"),
        description=meta.get("description"),
        og_title=meta.get("og_title"),
        og_description=meta.get("og_description"),
        og_site_name=meta.get("og_site_name"),
        transcript_text_preview=meta.get("transcript_preview"),
        raw_length=int(meta.get("raw_len") or 0),
        notes="Fetched transcript tab successfully.",
    )

@router.post("/fetch-meta", response_model=FetchMetaResponse)
@router.post("/fetch-meta/", response_model=FetchMetaResponse)
async def fetch_meta_post(body: FetchMetaBody):
    url = str(body.url)
    html = await fetch_transcript_html(url)
    meta = extract_meta(html)
    return FetchMetaResponse(
        url=url,
        title=meta.get("title"),
        description=meta.get("description"),
        og_title=meta.get("og_title"),
        og_description=meta.get("og_description"),
        og_site_name=meta.get("og_site_name"),
        transcript_text_preview=meta.get("transcript_preview"),
        raw_length=int(meta.get("raw_len") or 0),
        notes="Fetched transcript tab successfully.",
    )

@router.post("/fetch-chunk")
@router.post("/fetch-chunk/")
async def fetch_chunk_post(body: FetchChunkBody):
    """
    Accepts a single 'chunk' and processes it using the same packing rules.
    Returns the same style of logs as /process but for one chunk.
    """
    chunks = [body.chunk]
    max_chars = body.max_chars
    max_segments = body.max_segments

    logs = []
    total = len(chunks)
    done = 0
    for start, end, batch in pack_batches(chunks, max_chars=max_chars, max_segments=max_segments):
        batch_chars = sum(len(x) for x in batch)
        done = end
        logs.append({
            "processed_of_total": f"{done} of {total}",
            "this_batch_segments": (end - start),
            "this_batch_chars": batch_chars,
        })

    return {
        "total_chunks": total,
        "batches": logs,
        "message": "Chunk processed.",
    }

# Mount router at root and /api (supports both prefixes)
app.include_router(router)
app.include_router(router, prefix="/api")

@app.post("/process", response_model=ProcessResponse)
async def process_chunks(req: ProcessRequest):
    """
    Demonstration endpoint: packs incoming chunks by limits and returns logs.
    Replace the inner processing with your real connector/model call.
    """
    chunks = req.chunks or []
    total = len(chunks)
    logs: List[ProcessBatchLog] = []
    done = 0

    for start, end, batch in pack_batches(
        chunks, max_chars=req.max_chars, max_segments=req.max_segments
    ):
        # ---- Replace this block with your real processing call ----
        batch_chars = sum(len(x) for x in batch)
        done = end
        logs.append(
            ProcessBatchLog(
                processed_of_total=f"{done} of {total}",
                this_batch_segments=(end - start),
                this_batch_chars=batch_chars,
            )
        )
        await asyncio.sleep(0)  # yield control
        # -----------------------------------------------------------

    return ProcessResponse(
        total_chunks=total,
        batches=logs,
        message="All chunks processed." if done >= total else "Processing completed with caveats.",
    )

# =========================
# Entrypoint (Render)
# =========================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Keep workers=1 so there's only one shared browser
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)
