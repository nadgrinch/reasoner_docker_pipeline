# tiago_docker_pipeline
Repository of docker containers for running Reasoner on Tiago++.
## Core
There is docker-compose.yml, which start all three containers. In current version (05/2025) inside each container is just run bash, so user can go into them and run stuff from there
## Volumes
There are three folders `melodic_ws`, `noetic_ws` and `ros2_ws` which are imported to containers as volumes to `src/user_packages`. To depth 3 all packages inside these folders should be found and build at every startup of the container.

After tunning and debuging package and setling down that I should be part of the image, the package should be build at image level, so build on startup is not needed.
