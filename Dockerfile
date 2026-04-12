FROM python:3.12-slim AS builder

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
COPY src ./src

# Install dependencies from lockfile exactly.
# no-editable avoids editable metadata overhead in runtime image.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim

WORKDIR /app

# Copy only runtime environment and package source from builder.
COPY --from=builder /app/.venv /app/.venv
COPY src ./src

# Use the uv-managed virtual environment when running python.
ENV PATH="/app/.venv/bin:${PATH}"

# Judge usage examples:
#   docker build -t dut-trust-agent .
#   docker run --rm --env-file .env dut-trust-agent
ENTRYPOINT ["python", "-m", "devday_agent.main"]
