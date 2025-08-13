# syntax=docker/dockerfile:1
FROM node:20-alpine

# No build step needed: we already have dist JS + stub compiler
WORKDIR /srv
ENV PORT=8891

# Copy only what the brain needs at runtime
# daemon/dist + compiler/bin (where fabricc.mjs lives)
COPY brains/language/daemon/dist ./brains/language/daemon/dist
COPY brains/language/compiler/bin ./brains/language/compiler/bin

# Health endpoint (optional later), just run server
EXPOSE 8891
CMD ["node", "brains/language/daemon/dist/server.js"]
