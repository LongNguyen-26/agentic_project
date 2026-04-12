FROM python:3.12-slim

# Install tools needed to fetch and install uv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency metadata first for reproducible locked installs.
COPY pyproject.toml uv.lock README.md ./

# Copy the project source.
COPY . .

# Install dependencies from lockfile exactly.
RUN uv sync --frozen --no-dev

# Use the uv-managed virtual environment when running python.
ENV PATH="/app/.venv/bin:${PATH}"

# Judge usage examples:
#   docker build -t dut-trust-agent .
#   docker run --rm --env-file .env dut-trust-agent
ENTRYPOINT ["python", "main.py"]
