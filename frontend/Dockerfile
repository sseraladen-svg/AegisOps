FROM node:22-alpine
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173
# --host binds 0.0.0.0 so the port publish actually reaches the dev server.
CMD ["npm", "run", "dev", "--", "--host"]
