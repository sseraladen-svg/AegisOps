"""Engine/session factory for the telemetry SQLite database."""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DB_URL

engine = create_engine(DB_URL, future=True)
Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)


def get_session():
    return Session()


@contextmanager
def session_scope():
    """Commit on clean exit, roll back on error, always close.

    Scripts that write incidents use this so a crash mid-run can't leave the
    incidents table holding a half-written detector pass.
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
