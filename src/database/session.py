"""
Database Session Management and Lifecycle
"""
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.database.models import Base

# Ensure data directory exists
db_path = settings.project_root / "data"
db_path.mkdir(parents=True, exist_ok=True)

# Connection string
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False
)

# Enable foreign keys for SQLite
if "sqlite" in settings.database_url:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI and service dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
