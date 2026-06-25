"""Точка входа сервиса. Пока это «пустое кафе»: один эндпоинт /health,
по которому видно, что сервис запущен. Модель подключим на шаге 5.
"""
from fastapi import FastAPI

# app — главный объект приложения. Его по строке "app.main:app" найдёт uvicorn
# и будет держать запущенным; title попадёт в заголовок документации на /docs.
app = FastAPI(title="Marketplace satisfaction service")


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка живости сервиса.

    Декоратор @app.get("/health") связывает GET-запрос на адрес /health с этой
    функцией. Возвращаем обычный словарь — FastAPI сам превратит его в JSON
    {"status": "ok"}. По этому эндпоинту позже nginx, Docker и CI проверяют,
    что сервис вообще поднялся.
    """
    return {"status": "ok"}
