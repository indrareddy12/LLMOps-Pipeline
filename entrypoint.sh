#!/usr/bin/env bash
set -e
echo "Starting model server..."
# Optional model download/prep steps could be added here
exec gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 model_server:app