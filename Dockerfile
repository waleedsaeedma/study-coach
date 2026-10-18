# The recipe for the Study Coach image.
# Build:  docker build -t study-coach .
# Run:    see "Run it in Docker" in README.md

# 1. Start from Python 3.14 (the same version as .venv). "slim" = a small Linux without extras.
FROM python:3.14-slim

# 2. All our files live in /app inside the container.
WORKDIR /app

# 3. Install the packages first. Docker caches this step, so changing app code does not reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the code (.dockerignore keeps out .env, .venv and data/).
COPY . .

# 5. The live voice server (FastAPI) also listens on 0.0.0.0, so the browser outside the container can reach it.
#    The browser's address for it (LIVE_PUBLIC_URL) is set when the container starts, because it differs per machine.
ENV LIVE_HOST=0.0.0.0

# 6. Streamlit listens on 8501, the live voice server on 8000.
#    0.0.0.0 = accept connections from outside the container, not only from inside.
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
