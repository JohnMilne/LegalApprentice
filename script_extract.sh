/bin/bash

echo "Getting built artifact out of the container"
docker create --name temp_build ${SVC_NAME}_build:latest
docker cp temp_build:/var/www/dist ./dist
docker rm -f temp_build