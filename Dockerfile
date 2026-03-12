FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASM_HOST=0.0.0.0 \
    ASM_PORT=1217 \
    ASM_DB_PATH=/app/data/stock_monitor.db

WORKDIR /app

COPY requirements.txt ./
# 默认优先使用中国大陆镜像源，加快 Docker 构建时的依赖下载。
RUN pip install --no-cache-dir \
    -i ${PIP_INDEX_URL} \
    --trusted-host ${PIP_TRUSTED_HOST} \
    -r requirements.txt

COPY . ./
RUN chmod +x start.sh && mkdir -p /app/data

EXPOSE 1217

CMD ["./start.sh"]
