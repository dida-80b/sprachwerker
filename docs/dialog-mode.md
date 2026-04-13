# Dialog Mode

`sprachwerker` now treats `Dialog mit KI` as the main recording workflow.

## Goal

The dialog mode is meant to reduce the "what should I say now?" problem.

Flow:

1. Voxtral generates one short prompt text.
2. Piper speaks that prompt.
3. The user answers freely.
4. The user still marks sentence boundaries with the space bar.
5. Voxtral can rate each segment as `green`, `yellow` or `red`.
6. After a small round, ASR and review work exactly like before.

This keeps the existing review workflow intact and only changes how spoken material is collected.

## Current Backend Split

- Prompt generation: Voxtral via `llama-completion`
- Prompt playback: Piper via HTTP `POST /synthesize`
- Segment ASR: Voxtral via `llama-mtmd-cli`
- Segment quality rating: Voxtral via `llama-mtmd-cli`

The important part is that the system is used in phases:

1. generate prompt
2. play prompt
3. record user response
4. later run batch ASR

That avoids running multiple large language backends at the same time.

## Required Environment

Relevant variables:

```env
PIPER_URL=http://127.0.0.1:5150
PIPER_DEFAULT_VOICE=de_DE-thorsten-high
VOXTRAL_TEXT_CLI_PATH=/opt/llama-bin/llama-completion
VOXTRAL_PROMPT_MAX_TOKENS=48
VOXTRAL_PROMPT_TIMEOUT_SECONDS=90
```

`/health` now also reports:

- `text_cli_exists`
- `piper_ok`
- `piper_error`

## API Endpoints

The dialog UI uses these backend routes:

- `GET /dialog/voices`
- `POST /dialog/prompt`
- `POST /dialog/speak`

`/dialog/prompt` returns one short prompt text.

`/dialog/speak` proxies the text to Piper and returns WAV audio.

## Piper

`sprachwerker` does not bundle a Piper server internally.
It expects a reachable Piper HTTP service.

The existing Piper service from `anrufwerker/piper` works for this.

Expected routes:

- `GET /health`
- `GET /voices`
- `POST /synthesize`

## Notes

- Only microphone input should be recorded, never the TTS playback.
- The dialog mode is currently a lightweight V1.
- It is intentionally not a full multi-turn AI conversation system.
- The existing review, approval and dataset save flow stays the source of truth.
- The segment ampel is currently:
  - `green` = TTS + ASR
  - `yellow` = ASR only
  - `red` = discard
