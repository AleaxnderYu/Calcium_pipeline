#!/bin/bash
# Script to restart Open WebUI with follow-up questions disabled

echo "Stopping current Open WebUI container..."
docker stop open-webui

echo "Removing container (data will be preserved in volume)..."
docker rm open-webui

echo "Starting Open WebUI with updated configuration..."
docker run -d \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e ENABLE_COMMUNITY_SHARING=false \
  -e ENABLE_MESSAGE_RATING=false \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_SIGNUP=false \
  -e DEFAULT_MODELS="" \
  -e ENABLE_EVALUATION_ARENA_MODELS=false \
  -e SHOW_ADMIN_DETAILS=false \
  -e ENABLE_ADMIN_EXPORT=false \
  -e ENABLE_ADMIN_CHAT_ACCESS=false \
  -e ENABLE_TITLE_GENERATION=false \
  -e ENABLE_TAGS_GENERATION=false \
  -e ENABLE_SEARCH_QUERY=false \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

echo ""
echo "✓ Open WebUI restarted successfully!"
echo "  URL: http://localhost:3000"
echo "  Title generation: DISABLED"
echo "  Tags generation: DISABLED"
echo "  Search query: DISABLED"
echo ""
