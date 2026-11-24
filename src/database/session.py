"""Database Session and Connection Management."""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base
from config.settings import settings
from config.logging_config import logger

# SQLite specific config
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

def init_db():
    """Create all database tables on startup if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at {settings.DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e

@contextmanager
def get_db_session() -> Session:
    """Context manager for scoped database sessions with auto-rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rollback due to exception: {e}")
        raise e
    finally:
        session.close()

def get_db():
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
