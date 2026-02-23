"""Shared FastAPI dependencies used across route modules."""

from fastapi.security import HTTPBearer

from database import SessionLocal
from auth import get_current_agent_factory


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


security = HTTPBearer()

get_current_agent = get_current_agent_factory(get_db)
