# Free Recording Mode

The Free Recording tab is designed for speakers who find the space-bar workflow disruptive — particularly useful for dialect recordings, long takes, or anyone who just wants to talk naturally without managing boundaries manually.

## How It Works

1. Select an active dataset profile (same dropdown as the Recording tab).
2. Click **Aufnahme starten** — the browser requests microphone access and records continuously.
3. Speak freely. Make natural pauses between sentences; these become segment boundaries.
4. Click **Aufnahme stoppen** — the audio is sent to the backend.
5. The backend runs `ffmpeg silencedetect` and splits the recording at silence midpoints.
6. Segments appear as a list with inline audio players and editable transcript fields.
7. Click **Alle transkribieren** — Voxtral transcribes each segment sequentially.
8. Review and correct any transcripts directly inline.
9. Click **Alle speichern** — all segments with a non-empty transcript are saved to the active dataset.

## Silence Detection

The minimum silence duration is configurable in the UI (default: 700 ms). Increase it if your speech style includes longer pauses within a sentence; decrease it if you speak quickly with short inter-sentence pauses.

Backend parameters (via `POST /free-record/split`):

| Parameter | Default | Description |
|---|---|---|
| `min_silence_ms` | 700 | Minimum silence duration to trigger a split (milliseconds) |
| `silence_thresh_db` | -35.0 | Silence threshold in dBFS |

## Storage

Saved segments follow the same storage model as the Recording tab:

- Audio and text files in `output/_sprachwerker/<language>/<dataset>-<bucket>/`
- Manifest entry appended to `_review_manifest.jsonl`
- Duration bucket (`kurz`, `mittel`, `lang`, `sehr-lang`) assigned automatically from segment length

## Differences from Dialog Mode

| | Dialog Mode | Free Recording |
|---|---|---|
| Sentence boundaries | Manual (space bar) | Automatic (silence) |
| Prompting | LLM + Piper generates topic | None — speaker decides |
| Best for | Structured, emotion-conditioned takes | Long natural speech, dialect, bulk recording |
