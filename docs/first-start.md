# First Start

This guide is the shortest path from fresh clone to a working local `sprachwerker` instance with automatic `llama.cpp` build and automatic Voxtral download.

Live-tested on a ROCm machine on March 26, 2026.

## 1. Prepare `.env`

```bash
cp .env.example .env
```

Recommended defaults:

- ROCm host:
  - `LLAMA_BACKEND=rocm`
  - `ASR_BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete`
  - `LLAMA_BUILDER_BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete`
- CUDA host:
  - `LLAMA_BACKEND=cuda`
  - `ASR_BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  - `LLAMA_BUILDER_BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04`

Optional:

- Set `HF_TOKEN` if you use a gated or private Hugging Face repo.
- Leave `UI_DEFAULT_LANG` empty if you want `APP_REGION_GROUP` to decide between German and English.

## 2. Start The Stack

ROCm:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build -d
```

CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d
```

## 3. What Happens On First Start

The first start can take a while.

Important:

- `http://127.0.0.1:8095/` stays offline until both bootstrap jobs are done.
- `asr-api` can sit in `Created` state for a while. That is expected.
- The main Voxtral GGUF is large and can take many minutes to download.

Order of work:

1. Docker builds the runtime image and the `llama-builder` image.
2. `llama-builder` clones and compiles `llama.cpp`.
3. `model-fetcher` downloads the required Voxtral files from Hugging Face.
4. `asr-api` starts only after both steps completed successfully.

Persistent Docker volumes:

- `sprachwerker_llama-bin`
- `sprachwerker_llama-src`
- `sprachwerker_voxtral-models`

That means later restarts are much faster unless you force a rebuild or change model files.

## 4. Check Progress

Container state:

```bash
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep sprachwerker
```

Builder logs:

```bash
docker logs sprachwerker-llama-builder
```

Model download logs:

```bash
docker logs sprachwerker-model-fetcher
```

If the Hugging Face download looks quiet, that does not automatically mean it is stuck. Check whether the incomplete file keeps growing:

```bash
docker exec sprachwerker-model-fetcher sh -lc 'stat -c "%n %s %y" /models/.cache/huggingface/download/*.incomplete 2>/dev/null || true'
```

## 5. Verify Artifacts

Check built `llama.cpp` binaries:

```bash
docker run --rm -v sprachwerker_llama-bin:/artifacts ubuntu:22.04 ls -lah /artifacts
```

Check downloaded Voxtral files:

```bash
docker run --rm -v sprachwerker_voxtral-models:/models ubuntu:22.04 ls -lah /models
```

Expected files:

- `mistralai_Voxtral-Small-24B-2507-Q4_K_M.gguf`
- `mmproj-mistralai_Voxtral-Small-24B-2507-f16.gguf`

## 6. Verify The App

Once `sprachwerker` is running:

```bash
curl -fsS http://127.0.0.1:8095/health
```

Then open:

- UI: `http://127.0.0.1:8095/`
- Health: `http://127.0.0.1:8095/health`

## 7. Common Issues

`sprachwerker` stays in `Created`:

- Usually `model-fetcher` is still downloading or `llama-builder` has not completed yet.
- Check both containers first.

`llama-builder` exits with error:

- ROCm or CUDA container access is not configured correctly for the host.
- The selected base image does not match the backend.
- Upstream `llama.cpp` build flags changed.

`model-fetcher` fails:

- Wrong repo or file names in `.env`
- Gated model repo without `HF_TOKEN`
- Temporary network or Hugging Face rate limit issue

`/health` returns `ok` but `model_exists=false`:

- The model volume does not contain the expected file names.
- `VOXTRAL_MODEL_FILENAME` or `VOXTRAL_MMPROJ_FILENAME` do not match the real files.

Inference starts but crashes:

- Backend mismatch between built `llama.cpp` and the host GPU stack
- `VOXTRAL_GPU_LAYERS` is too aggressive for the available VRAM

## 8. Force A Fresh Build

To force a fresh `llama.cpp` build:

```env
LLAMA_FORCE_REBUILD=1
```

Then rerun:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build -d
```

If you want a fully clean bootstrap, also remove the named volumes first:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml down
docker volume rm sprachwerker_llama-bin sprachwerker_llama-src sprachwerker_voxtral-models
```
