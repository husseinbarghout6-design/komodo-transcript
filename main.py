from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from playwright.async_api import async_playwright
import re

app = FastAPI(title="Komodo Transcript Fetcher")

class Req(BaseModel):
    url: HttpUrl

# We return OK here so Render can health-check the service
@app.get("/")
def health():
    return {"status": "ok"}

# Common selectors to try for the transcript container
TRANSCRIPT_SELECTORS = [
    '[data-testid="transcript"]',
    '[id*="transcript"]',
    '[role="tabpanel"]:has-text("Transcript")',
    'section:has(h2:regexp("^\\s*Transcript\\s*$"))',
]

async def extract_transcript(page):
    # Try clicking/activating a Transcript tab if present
    for locator in ["text=Transcript", "role=tab[name='Transcript']"]:
        if await page.locator(locator).count():
            await page.click(locator)
            break

    # Wait and read likely containers
    for sel in TRANSCRIPT_SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=6000)
            text = await page.locator(sel).inner_text()
            if text and len(text.strip()) > 10:
                return text
        except Exception:
            pass

    # Fallback: grab the whole page’s visible text (heuristic)
    return await page.locator("body").inner_text()

def normalize_lines_to_segments(text: str):
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
            mm, ss = ts.split(":")
            try:
                sec = int(mm) * 60 + float(ss)
            except ValueError:
                sec = None
            segments.append({"start": sec, "text": rest.strip() if rest else ""})
        else:
            segments.append({"text": ln})
    return segments

@app.post("/api/fetch-transcript")
async def fetch_transcript(req: Req):
    """
    Body: { "url": "https://komododecks.com/recordings/xxxx?tab=transcript" }
    Returns: { "title": "...", "transcript": [ {start: seconds?, text: "..."} ] }
    """
    async with async_playwright() as p:
        # Launch Chromium headless; --no-sandbox is required on many hosts
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        try:
            await page.goto(str(req.url), wait_until="domcontentloaded", timeout=30000)

            # If URL lacks ?tab=transcript, try appending it
            if "tab=transcript" not in str(req.url):
                joiner = "&" if "?" in str(req.url) else "?"
                await page.goto(
                    str(req.url) + f"{joiner}tab=transcript",
                    wait_until="domcontentloaded",
                    timeout=15000
                )

            text = await extract_transcript(page)
            title = await page.title()
        finally:
            await browser.close()

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=404, detail="Transcript not found or too short (is the page public?)")

    return {"title": title, "transcript": normalize_lines_to_segments(text)}
