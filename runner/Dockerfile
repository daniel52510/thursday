# THURSDAY's self-improvement sandbox (runner/agent.py). No secrets, no Docker socket, not root.
# Built from the repo root so it gets core's pinned dependencies: tests/ import the real modules.
# runner/Dockerfile.dockerignore lets in only the requirements files and runner/: never .env.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-test.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-test.txt \
    && useradd -m -u 1000 runner && mkdir /work /out && chown runner /work /out
COPY runner/agent.py runner/checks.py runner/codeindex.py /runner/
USER runner
# /repo.git is the host repo's .git, mounted read-only and owned by someone else.
RUN git config --global --add safe.directory '*'
ENTRYPOINT ["python", "-u", "/runner/agent.py"]
