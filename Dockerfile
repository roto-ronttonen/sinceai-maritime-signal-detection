# Use Python 3.13 slim image
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app


COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY *.py ./



# Set entrypoint to run main.py with uv
ENTRYPOINT ["uv", "run", "main.py"]