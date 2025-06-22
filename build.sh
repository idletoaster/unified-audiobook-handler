#!/bin/bash
# Build script for unified audiobook handler container
# This script builds and pushes the container to GitHub Container Registry

echo "Building unified-audiobook-handler container..."

# Build the container
docker build -t ghcr.io/idletoaster/unified-audiobook-handler:latest .

# Login to GitHub Container Registry (requires GITHUB_TOKEN)
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Push the container
docker push ghcr.io/idletoaster/unified-audiobook-handler:latest

echo "Container published to ghcr.io/idletoaster/unified-audiobook-handler:latest"
echo "Container is now publicly accessible for RunPod"
