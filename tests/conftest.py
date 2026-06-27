"""Общие фикстуры для тестов. Файл conftest.py pytest подхватывает сам — импортировать не нужно."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """HTTP-клиент к приложению, поднятому в памяти (без реального сервера).

    scope="session": клиент создаётся один раз на весь прогон. При входе в
    `with TestClient(app)` запускается lifespan — то есть грузится модель; делать
    это на каждый тест было бы расточительно. yield отдаёт клиент тестам, а на
    выходе приложение корректно гасится.
    """
    with TestClient(app) as c:
        yield c