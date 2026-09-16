import psycopg

from app.config import settings


def get_connection():
    return psycopg.connect(
        settings.dsn,
        connect_timeout=10,
    )