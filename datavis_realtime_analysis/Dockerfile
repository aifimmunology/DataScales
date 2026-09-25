# Stage 1: Build
FROM oven/bun:1 AS builder

WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install

COPY tsconfig.json vite.config.ts index.html ./
COPY public ./public
COPY src ./src

RUN bun run build

# Stage 2: Production
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

# 127.0.0.1, not localhost: in-container localhost resolves to [::1] but nginx
# only listens on IPv4, so the check always refused and flagged unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD wget -q --spider http://127.0.0.1:80 || exit 1
