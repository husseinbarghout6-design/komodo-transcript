# ✅ Playwright + Python + Chromium preinstalled
FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install only what we need (no playwright here; base image already has it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

EXPOSE 8080

# Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
