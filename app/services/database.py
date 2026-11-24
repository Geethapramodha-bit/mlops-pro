import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Use the Render persistent disk when DATA_DIR is configured.
# Otherwise, retain the local development data directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{(DATA_DIR / 'botplayer.db').as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
