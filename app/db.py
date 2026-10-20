"""Job records and idempotency keys (SQLite by default, Postgres via DATABASE_URL)."""
import json
import threading
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _now():
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(40), primary_key=True)
    kind = Column(String(32), default="broll")
    status = Column(String(16), default="queued")  # queued | running | succeeded | failed
    progress = Column(Integer, default=0)
    stage = Column(String(64), default="queued")
    request_json = Column(Text)
    result_json = Column(Text, nullable=True)
    error_json = Column(Text, nullable=True)
    idempotency_key = Column(String(255), unique=True, nullable=True)
    request_hash = Column(String(64), nullable=True)
    correlation_id = Column(String(128), nullable=True)
    parent_job_id = Column(String(40), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


_engine = None
_Session = None
_lock = threading.Lock()


def init_db():
    global _engine, _Session
    with _lock:
        if _engine is None:
            url = get_settings().db_url
            kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
            _engine = create_engine(url, **kw)
            Base.metadata.create_all(_engine)
            _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def session():
    init_db()
    return _Session()


def create_job(request: dict, idem_key: str | None, req_hash: str, correlation_id: str | None,
               kind: str = "broll", parent: str | None = None) -> tuple[Job, bool]:
    """Returns (job, created). Same idempotency key returns the existing job."""
    with session() as s:
        if idem_key:
            existing = s.scalar(select(Job).where(Job.idempotency_key == idem_key))
            if existing:
                return existing, False
        job = Job(id="job_" + uuid.uuid4().hex[:20], kind=kind, request_json=json.dumps(request),
                  idempotency_key=idem_key, request_hash=req_hash, correlation_id=correlation_id,
                  parent_job_id=parent)
        s.add(job)
        try:
            s.commit()
        except IntegrityError:  # two identical requests raced: return the winner
            s.rollback()
            return s.scalar(select(Job).where(Job.idempotency_key == idem_key)), False
        return job, True


def get_job(job_id: str) -> Job | None:
    with session() as s:
        return s.get(Job, job_id)


def update_job(job_id: str, **fields) -> None:
    with session() as s:
        job = s.get(Job, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, json.dumps(v) if k in ("result_json", "error_json") and not isinstance(v, str) and v is not None else v)
        s.commit()


def jobs_with_status(status: str) -> list[Job]:
    with session() as s:
        return list(s.scalars(select(Job).where(Job.status == status)))


def db_ok() -> bool:
    try:
        with session() as s:
            s.execute(select(1))
        return True
    except Exception:
        return False
