from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import chat, web_search, workflow

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="MINDIGENOUS Core", version="0.1.0")
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend_origin, "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api")
    app.include_router(workflow.router, prefix="/api")
    app.include_router(web_search.router, prefix="/api")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
