# Bootstrap And Backends

This document explains how `sprachwerker` now bootstraps both `llama.cpp` and the Voxtral model stack.

## Overview

The stack is split into three services:

- `llama-builder`
- `model-fetcher`
- `asr-api`

Startup order:

1. `llama-builder` clones and builds `llama.cpp`
2. `model-fetcher` downloads the required Voxtral GGUF files
3. `asr-api` starts after both completed successfully

Artifacts are stored in Docker volumes:

- `sprachwerker_llama-bin`
- `sprachwerker_llama-src`
- `sprachwerker_voxtral-models`

This avoids host-specific binary folders and makes the repo easier to publish.

## Compose Files

Base:

- `docker-compose.yml`

ROCm override:

- `docker-compose.rocm.yml`

CUDA override:

- `docker-compose.cuda.yml`

Use ROCm:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build -d
```

Use CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d
```

## Environment Variables

Core builder variables:

- `LLAMA_BACKEND`
- `ASR_BASE_IMAGE`
- `LLAMA_BUILDER_BASE_IMAGE`
- `LLAMA_REPO`
- `LLAMA_REF`
- `LLAMA_FORCE_REBUILD`
- `AMDGPU_TARGETS`
- `CUDA_DOCKER_ARCH`
- `LLAMA_CMAKE_EXTRA_ARGS`

Model download variables:

- `HF_TOKEN`
- `VOXTRAL_HF_REPO`
- `VOXTRAL_MODEL_FILENAME`
- `VOXTRAL_MMPROJ_FILENAME`

## Recommended Values

ROCm:

```env
LLAMA_BACKEND=rocm
ASR_BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete
LLAMA_BUILDER_BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete
```

CUDA:

```env
LLAMA_BACKEND=cuda
ASR_BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
LLAMA_BUILDER_BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
```

CPU-only:

```env
LLAMA_BACKEND=cpu
ASR_BASE_IMAGE=ubuntu:22.04
LLAMA_BUILDER_BASE_IMAGE=ubuntu:22.04
```

## HF Token

`HF_TOKEN` is optional for public Hugging Face repos.

It is required when:

- the model repo is gated
- the model repo is private
- rate limits make anonymous download impractical

Current default source in `.env.example` is public:

- `bartowski/mistralai_Voxtral-Small-24B-2507-GGUF`

## Inspecting Bootstrap State

Builder status:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml ps
```

Builder logs:

```bash
docker logs sprachwerker-llama-builder
docker logs sprachwerker-model-fetcher
```

List created volumes:

```bash
docker volume ls | grep sprachwerker
```

Inspect `llama.cpp` artifacts:

```bash
docker run --rm -v sprachwerker_llama-bin:/artifacts ubuntu:22.04 ls -la /artifacts
```

Inspect Voxtral model files:

```bash
docker run --rm -v sprachwerker_voxtral-models:/models ubuntu:22.04 ls -lah /models
```

## Force Rebuild

If `llama.cpp` should be rebuilt from scratch:

```env
LLAMA_FORCE_REBUILD=1
```

Then rerun compose:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build -d
```

## Common Failure Modes

`llama-builder` fails early:

- wrong backend image for the selected backend
- missing ROCm/CUDA runtime access
- upstream `llama.cpp` build flags changed

`model-fetcher` fails:

- repo or filename mismatch
- gated repo without `HF_TOKEN`
- Hugging Face rate limit or network issue

`asr-api` health is `ok` but `model_exists=false`:

- model-fetcher did not populate the `voxtral-models` volume
- filenames in `.env` do not match downloaded files

`asr-api` starts but inference crashes:

- backend mismatch between built `llama.cpp` and host GPU stack
- too aggressive `VOXTRAL_GPU_LAYERS`
- incompatible ROCm/CUDA image for the host driver version

## Notes

- ROCm and CUDA are intentionally separated by compose override files.
- This is more reliable than trying to auto-detect GPU backend inside one ambiguous container definition.
- The base compose file is backend-neutral and only defines the bootstrap flow.
