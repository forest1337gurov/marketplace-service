# Базовый образ: Python 3.11 в минимальной (slim) редакции
FROM python:3.11-slim

# LightGBM требует системную библиотеку OpenMP (libgomp), без неё модель не импортируется
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Внутри контейнера работаем из /app — как `cd /app` в терминале
WORKDIR /app

# Зависимости отдельным слоем — Docker закеширует и не будет переустанавливать при правках кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код и обученная модель. Модель последней — чтобы её обновление не сбивало кеш зависимостей
COPY app/ ./app/
COPY models/ ./models/

# Не буферизовать stdout — иначе логи могут «застрять» и не попасть в docker logs
ENV PYTHONUNBUFFERED=1
# Документируем, что контейнер слушает порт 8000
EXPOSE 8000

# --host 0.0.0.0 обязателен: иначе снаружи контейнера сервис недоступен
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]