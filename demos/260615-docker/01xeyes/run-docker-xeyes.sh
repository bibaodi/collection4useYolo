#!/bin/bash

# build image
CMD="docker build -t xeyes01 ."
echo "build image script:${CMD}"
# this is critical
xhost +local:docker

# both works
## 01
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix xeyes01

function run_xeyes01(){
## 02
docker run -it --rm \
  -e DISPLAY=:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  xeyes01   # or any X11 app image
non-network local connections being added to access control list
}

## Dockerfile
function createdockerfile(){
cat > Dockerfile << 'EOF'
FROM alpine:latest
RUN apk add --no-cache xeyes
CMD ["xeyes"]
EOF
}
