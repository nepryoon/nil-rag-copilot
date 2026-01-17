from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time

app = FastAPI(title="NIL RAG Copilot API", version=os.getenv("APP_VERSION", "0.0.1"))

allowed = os.getenv("ALLOWED_ORIGINS", "https://www.neuromorphicinference.com")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

BOOT_TS = int(time.time())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {
        "app": "nil-rag-copilot-api",
        "version": os.getenv("APP_VERSION", "0.0.1"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "boot_ts": BOOT_TS,
    }
