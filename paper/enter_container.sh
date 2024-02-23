#!/usr/bin/bash

docker compose up -d
xhost +local:
docker compose exec latex bash
xhost -local:
