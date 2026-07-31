# Stage 1: Build
FROM oven/bun:1 AS builder

WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install

COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src

RUN bun run build

# Stage 2: Production
FROM oven/bun:1-alpine

WORKDIR /app

COPY --from=builder /app/dist ./dist

RUN bun add serve

CMD ["bunx", "serve", "dist", "-p", "3000"]

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD bun -e "fetch('http://localhost:3000').then(r => process.exit(r.ok ? 0 : 1))"
