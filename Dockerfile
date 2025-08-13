FROM python:3.11-slim

# System deps needed by Chromium
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxdamage1 libxrandr2 libgbm1 libasound2 libatspi2.0-0 \
    fonts-liberation libjpeg62-turbo libxshmfence1 xvfb wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install --with-deps chromium

COPY . .

# Render injects PORT; default to 8080 for local runs
ENV PORT=8080
EXPOSE 8080

CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
