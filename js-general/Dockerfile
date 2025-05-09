FROM node:20 as builder

WORKDIR /usr/app
COPY package*.json ./
RUN npm install && npm cache clean --force
RUN apt-get update && \
    apt-get install -y ffmpeg 
COPY . .

FROM node:20
COPY --from=builder /usr/app/ /