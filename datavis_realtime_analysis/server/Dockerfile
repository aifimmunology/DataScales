FROM ghcr.io/prefix-dev/pixi:0.67.0

WORKDIR /app

COPY server/pixi.toml server/pixi.lock ./
RUN pixi install --locked

COPY server ./server
COPY gpu ./gpu

EXPOSE 8000

CMD ["pixi", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
