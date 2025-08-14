# Language Brain (gRPC) daemon
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --only=prod || npm install --omit=dev
COPY proto ./proto
COPY server ./server
COPY brains ./brains
COPY node_modules/fabric-fab-guard-cli ./node_modules/fabric-fab-guard-cli
ENV PORT=8891 FAB_ROOT=/app
EXPOSE 8891
CMD ["node","server/index.js"]
