# BUILD & RUN
FROM node:22.20.0-alpine AS build-frontend
WORKDIR /app
COPY . .
RUN npm install && npm run build

FROM nginx:alpine
WORKDIR /app
COPY --from=build-frontend /app/dist /usr/share/nginx/html
COPY default.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]