# Multi-stage Dockerfile optimized for production
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip wheel -r requirements.txt -w /wheels

FROM python:3.11-slim
WORKDIR /app
RUN useradd -m appuser
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY . .
RUN chmod +x /app/entrypoint.sh
USER appuser
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]