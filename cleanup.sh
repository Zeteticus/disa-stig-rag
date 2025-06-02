#!/bin/bash

podman stop -a
podman system prune -a
podman rmi -a -f
podman volume rm -a -f
podman pod prune -f
systemctl --user stop podman
sleep 2
systemctl --user start podman
