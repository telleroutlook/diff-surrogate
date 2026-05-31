FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/diff-surrogate

COPY pyproject.toml README.md ./
COPY diff_surrogate/ diff_surrogate/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dev]"

COPY Makefile .
COPY benchmarks/ benchmarks/
COPY tests/ tests/

ENV PYTHONHASHSEED=42

CMD ["make", "test"]
