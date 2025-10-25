#!/bin/bash
# Build Docker base image for calcium imaging analysis

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Building Calcium Imaging Docker Image${NC}"
echo -e "${BLUE}================================${NC}"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Image name
IMAGE_NAME="${1:-calcium_imaging:latest}"

echo -e "${GREEN}Building image: ${IMAGE_NAME}${NC}"
echo

# Build image
docker build \
    -f Dockerfile.calcium_imaging \
    -t "$IMAGE_NAME" \
    .

echo
echo -e "${GREEN}✓ Image built successfully!${NC}"
echo
echo "Image details:"
docker images "$IMAGE_NAME"

echo
echo -e "${BLUE}To use this image:${NC}"
echo "  1. Set in .env: EXECUTOR_BACKEND=docker"
echo "  2. Set in .env: DOCKER_BASE_IMAGE=$IMAGE_NAME"
echo "  3. Run: python main.py --request \"...\" --images ./data/images"
echo
echo -e "${GREEN}Done!${NC}"
