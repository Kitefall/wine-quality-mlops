FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

RUN pip install --no-cache-dir dvc-s3

COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "src/app/api.py"]
