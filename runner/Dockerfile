# THURSDAY's self-improvement sandbox (runner/agent.py). No secrets, no Docker socket, not root.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir httpx==0.28.1 \
    && useradd -m -u 1000 runner && mkdir /work /out && chown runner /work /out
COPY agent.py checks.py /runner/
USER runner
# /repo.git is the host repo's .git, mounted read-only and owned by someone else.
RUN git config --global --add safe.directory '*'
ENTRYPOINT ["python", "-u", "/runner/agent.py"]
