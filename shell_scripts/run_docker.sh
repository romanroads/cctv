DOCKER_IMAGE_NAME="696706365164.dkr.ecr.us-west-2.amazonaws.com/cctv:latest"
CONDA_ENV_NAME="element_ubuntu"
JOB_NAME_INSTALL_DEP="job_0"

docker run --gpus all --name "${JOB_NAME_INSTALL_DEP}" -e NVIDIA_VISIBLE_DEVICES=0 -i --rm \
-v "$PWD/data":/cctv/data \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v "$PWD/credentials":/cctv/credentials \
"${DOCKER_IMAGE_NAME}" \
/bin/bash -c ". activate ${CONDA_ENV_NAME} && cd python && ./shell_scripts/run_det.sh"