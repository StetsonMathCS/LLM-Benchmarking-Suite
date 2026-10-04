FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm g++ cppcheck \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest==8.3.5 pylint==3.3.4 bandit==1.8.3
RUN npm install --global eslint@9.20.1
RUN useradd --create-home --uid 10001 facets
USER facets
WORKDIR /work
