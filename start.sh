#!/bin/sh
# Starts the container: nginx in the background, Streamlit in the front.
# (The live voice server is started by the app itself, when you choose "Live call".)
# If Streamlit stops, the container stops too, so Docker or Hugging Face can restart it.
set -e
nginx -e /dev/stderr -c /app/nginx.conf      # -e: log to the screen (a normal user can't write /var/log)
exec streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true
