FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev || npm install --only=prod
COPY proto ./proto
COPY server ./server
COPY brains ./brains
EXPOSE 8891
CMD ["node","server/index.js","--bind","0.0.0.0:8891"]