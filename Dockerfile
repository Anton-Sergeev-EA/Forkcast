# Воспроизводимое окружение Forkcast
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .
# docker run --rm -v "$PWD/reports:/app/reports" forkcast run
ENTRYPOINT ["forkcast"]
CMD ["run"]
