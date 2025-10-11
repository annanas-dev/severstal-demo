FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv

RUN uv sync --frozen

COPY *.py ./

EXPOSE 8502

CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=8502", "--server.address=0.0.0.0"]

