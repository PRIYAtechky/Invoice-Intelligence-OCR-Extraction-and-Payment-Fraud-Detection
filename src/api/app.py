"""FastAPI Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.database.session import init_db
from config.settings import settings
from config.logging_config import logger

def create_app() -> FastAPI:
    app = FastAPI(
        title=f"{settings.APP_NAME} API",
        version=settings.APP_VERSION,
        description="Production REST API for Invoice Intelligence: OCR extraction, multi-rule data validation, evidence-based payment fraud detection, and reviewer workflows.",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 router
    app.include_router(router)

    @app.on_event("startup")
    def on_startup():
        logger.info("Initializing Invoice Intelligence database...")
        init_db()

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
