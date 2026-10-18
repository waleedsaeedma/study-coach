# The recipe for the Study Coach image.
# Build:  docker build -t study-coach .
# Run:    see "Run it in Docker" in README.md

# 1. Start from Python 3.14 (the same version as .venv). "slim" = a small Linux without extras.
FROM python:3.14-slim

# 2. nginx = the "doorman": one public port, it sends /live/ to the voice server and the rest to Streamlit.
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*

# 3. A normal user (number 1000), not root. Hugging Face runs containers as this user, so we test the same way.
RUN useradd -m -u 1000 user

# 4. All our files live in /app inside the container.
WORKDIR /app

# 5. Install the packages first. Docker caches this step, so changing app code does not reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the code (.dockerignore keeps out .env, .venv and data/), owned by our user.
#    data/ must exist and be writable: notes, Chroma and memory are saved there.
COPY --chown=user . .
RUN mkdir -p data && chown user data
USER user
ENV HOME=/home/user

# 7. The browser opens the live call at /live (through nginx), on the same address as the app.
ENV LIVE_PUBLIC_URL=/live

# 8. One public port: nginx listens on 7860 (Hugging Face's default). Streamlit (8501) and the voice
#    server (8000) only listen inside the container.
EXPOSE 7860
CMD ["sh", "start.sh"]
