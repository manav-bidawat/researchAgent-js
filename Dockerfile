# syntax=docker/dockerfile:1
#
# The TypeScript layer (CLI, Express demo server, MCP server) is compiled in a Node
# stage; the runtime image is Python, because the engine in src/ runs there, with the
# Node 22 binary copied in to drive it through bridge/worker.py.

# --- build: compile TypeScript to dist/ -------------------------------------------
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json tsconfig.json ./
RUN npm ci
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
RUN npm run build

# --- deps: production node_modules only -------------------------------------------
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- runtime: Python engine + Node ------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# package.json marks the repository root for app/paths.ts.
COPY package.json ./
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY bridge ./bridge
COPY src ./src
COPY config.yaml ./
COPY web/static ./web/static
# `eval` reads the question set from here and writes to eval/results/.
COPY eval/questions.json ./eval/questions.json

ENV SCIAGENT_PYTHON=python3 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["node", "dist/app/cli.js", "serve", "--host", "0.0.0.0"]
