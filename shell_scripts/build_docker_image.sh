DOCKER_IMAGE_NAME="696706365164.dkr.ecr.us-west-2.amazonaws.com/cctv:latest"

docker build -t "${DOCKER_IMAGE_NAME}" -f ./docker/Dockerfile .
