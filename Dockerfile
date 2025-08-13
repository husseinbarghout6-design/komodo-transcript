# Uses Playwright + Python preinstalled (v1.46.0). Jammy = Ubuntu 22.04
FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

# Prevent Python from writing pyc files & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Only copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install deps (uvicorn/fastapi/pydantic already fine)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Health port exposure (Render sets $PORT at runtime)
EXPOSE 8080

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
