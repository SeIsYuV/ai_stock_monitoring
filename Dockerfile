FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASM_HOST=0.0.0.0 \
    ASM_PORT=1217 \
    ASM_DB_PATH=/app/data/stock_monitor.db

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN chmod +x start.sh && mkdir -p /app/data

EXPOSE 1217

CMD ["./start.sh"]
