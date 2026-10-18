"""Put the Study Coach online on Hugging Face Spaces.

Once:  hf auth login      (paste a token with "Write" access from huggingface.co/settings/tokens)
       add SPACE_APP_PASSWORD=... to .env (the password of the online app)
Then:  python deploy_hf.py

It uploads the last commit (never .env, data/ or the demo video), gives the Space your API keys as secrets,
and Hugging Face builds the Dockerfile. Run it again after every commit you want online.
"""
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = Path(__file__).resolve().parent
SPACE_NAME = "study-coach"
SECRETS = ["OPENAI_API_KEY", "TAVILY_API_KEY"]          # copied from .env to the Space's secret settings
SKIP = ("docs/", "README.md", ".gitattributes")         # the Space gets its own README and .gitattributes

# Hugging Face reads the settings at the top of the Space's README: run the Dockerfile, the app is on port 7860
SPACE_README = """---
title: Study Coach
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

An AI study tutor built with LangGraph. Code and docs: https://github.com/waleedsaeedma/study-coach
"""


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def committed_files() -> dict[str, bytes]:
    """{path: content} of every file in the last commit, exactly as Git stores it (Linux line endings)."""
    paths = git("ls-files").decode().splitlines()
    return {p: git("show", f"HEAD:{p}") for p in paths if not p.startswith(SKIP)}


def main():
    env = dotenv_values(ROOT / ".env")
    missing = [key for key in SECRETS + ["SPACE_APP_PASSWORD"] if not env.get(key)]
    if missing:
        raise SystemExit(f"Add these to .env first: {', '.join(missing)}")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("Note: you have changes that are not committed. Only the last commit is uploaded.")

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # secrets first, so the build that starts after the upload already has them
    for key in SECRETS:
        api.add_space_secret(repo_id, key, env[key])
    api.add_space_secret(repo_id, "APP_PASSWORD", env["SPACE_APP_PASSWORD"])   # the app is public: password on

    files = committed_files()
    operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=content) for p, content in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=SPACE_README.encode()))
    # files deleted here since the last upload are deleted in the Space too
    keep = set(files) | {"README.md", ".gitattributes"}
    old = api.list_repo_files(repo_id, repo_type="space")
    operations += [CommitOperationDelete(path_in_repo=p) for p in old if p not in keep]

    commit = git("rev-parse", "--short", "HEAD").decode().strip()
    api.create_commit(repo_id, repo_type="space", operations=operations, commit_message=f"Deploy {commit}")
    print(f"Uploaded commit {commit}. Hugging Face is building it now (a few minutes).")
    print(f"Build log: https://huggingface.co/spaces/{repo_id}")
    print(f"Your app:  https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
