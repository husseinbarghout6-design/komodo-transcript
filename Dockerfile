# ✅ Playwright + Python + Chromium already installed (v1.46.0, Ubuntu 22.04)
FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install only your Python deps (do NOT reinstall playwright here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add your app code
COPY . .

EXPOSE 8080

# Start FastAPI via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
