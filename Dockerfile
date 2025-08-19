FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

# Prevent .pyc files and ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Render port
EXPOSE 8080

# Use bash -lc to ensure env vars are loaded in container
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
