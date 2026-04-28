FROM node:20-alpine AS builder

WORKDIR /app
ENV NODE_ENV=production \
    REACT_APP_BACKEND_URL=http://localhost:8000

COPY package.json package-lock.json* ./
RUN npm install --silent

COPY public/    ./public/
COPY src/       ./src/
COPY tailwind.config.js ./
COPY postcss.config.js  ./

RUN npm run build

FROM nginx:1.25-alpine AS runtime

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/build /usr/share/nginx/html

RUN addgroup -S appgroup && adduser -S appuser -G appgroup
RUN chown -R appuser:appgroup /usr/share/nginx/html \
    && chown -R appuser:appgroup /var/cache/nginx \
    && chown -R appuser:appgroup /var/log/nginx \
    && touch /var/run/nginx.pid \
    && chown appuser:appgroup /var/run/nginx.pid

USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:3000 || exit 1

CMD ["nginx", "-g", "daemon off;"]