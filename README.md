# Unified Audiobook Handler

Phase 2 RunPod containerized audiobook converter with automatic engine selection for English→English (Coqui), English→Indian languages (IndicTrans2+AI4Bharat), and multilingual support.

## Quick Setup for RunPod

Use this public image directly in RunPod:
```
ghcr.io/idletoaster/unified-audiobook-handler:latest
```

## Features
- English→English conversion (Coqui TTS)
- English→Hindi/Telugu/Tamil/Bengali (IndicTrans2 + AI4Bharat TTS)  
- Spanish/French→Same language support
- GPU-accelerated processing
- Base64 PDF input/output
- Automatic engine selection

## RunPod Configuration
1. **Container Image**: `ghcr.io/idletoaster/unified-audiobook-handler:latest`
2. **GPU**: 24GB (RTX 4090) or higher
3. **Container Registry**: Public (no auth needed)
4. **Timeout**: 300 seconds

## GitHub Actions Auto-Build
This repository uses GitHub Actions to automatically build and publish the container to GitHub Container Registry on every push to main branch.

## Usage
The container accepts PDF data in base64 format and returns MP3 audio files for audiobook conversion across multiple languages.
