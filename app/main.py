from fastapi import FastAPI

from app.api.routes import router as copilot_router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "FastAPI backend for an AI-powered system design copilot. "
        "This first slice handles OpenAI-based orchestration only."
    ),
)

app.include_router(copilot_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
