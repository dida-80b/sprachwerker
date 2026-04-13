# Manifest Schema

`Sprachwerker` stores reviewed items as JSON Lines in:

```text
output/_sprachwerker/<language>/<dataset-bucket>/_review_manifest.jsonl
```

Each line is a single JSON object describing one approved segment. Audio and transcript files live next to the manifest entry and share the same numeric ID stem.

## Storage Rules

- One line per approved item
- UTF-8 encoded JSONL
- Audio file and text file share the same `id`
- `base_dataset` is the dataset name before bucket suffixing
- `bucket` reflects the duration bucket used for storage routing
- Workflow fields are intentionally generic so later exporters can target Kokoro, faster-whisper, or other pipelines
- Emotion stays separate from dialect and audio-domain metadata so TTS exports can filter delivery without losing language-variety labels

## Example

```json
{
  "id": "0000000042",
  "text": "Servus, i brauch an Termin fuer naechste Woche.",
  "raw_text": "servus i brauch an termin für nächste woche",
  "start": 12.48,
  "end": 16.91,
  "duration_seconds": 4.43,
  "bucket": "mittel",
  "base_dataset": "bayern-callcenter-seed",
  "task_mode": "tts",
  "audio_domain": "clean",
  "target_engine": "faster-whisper",
  "speaker_profile": "single_speaker",
  "dialect_label": "bairisch-niederbayern",
  "style_label": "neutral",
  "emotion_label": "drunk",
  "recording_quality": "clean",
  "tts_suitability": "needs_review"
}
```

## Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | string | yes | Zero-padded segment ID, shared by audio and text filenames. |
| `text` | string | yes | Reviewed text approved by the user. |
| `raw_text` | string | no | Unreviewed or original transcript before correction. |
| `start` | number or null | no | Segment start time in seconds relative to the source recording. |
| `end` | number or null | no | Segment end time in seconds relative to the source recording. |
| `duration_seconds` | number | yes | Segment duration in seconds. |
| `bucket` | string | yes | Duration bucket used for dataset routing. Allowed: `kurz`, `mittel`, `lang`, `sehr-lang`, `zu-lang`. |
| `base_dataset` | string | yes | Logical dataset name before the bucket suffix is appended. |
| `task_mode` | string | yes | Intended collection mode. Allowed: `tts`, `asr`, `both`. |
| `audio_domain` | string | yes | Audio environment or channel. Allowed internally: `clean`, `telephone`, `noisy`. The studio UI currently records only `clean`; `telephone` is intended for later imports such as Anrufwerker. |
| `target_engine` | string | yes | Primary downstream target. Current UI options: `kokoro`, `whisper`, `faster-whisper`, `custom`, optional `piper` if enabled later. |
| `speaker_profile` | string | yes | Speaker consistency profile. Allowed: `single_speaker`, `multi_speaker`. In the current UI, `multi_speaker` is only allowed for `task_mode=asr`. |
| `dialect_label` | string | no | Free-form dialect or accent descriptor such as `bairisch-oberpfalz`, `gsw-zuerich`, or `standarddeutsch`. |
| `style_label` | string | yes | Compatibility field. The current UI does not expose separate style tagging and writes `neutral` by default. |
| `emotion_label` | string | yes | Thorsten/Piper-style emotion tag. Allowed: `happy`, `angry`, `disgusted`, `drunk`, `neutral`, `sleepy`, `surprised`, `whispering`. |
| `recording_quality` | string | yes | Review-grade recording quality. Allowed: `clean`, `usable`, `noisy`. |
| `tts_suitability` | string | yes | Whether the clip should be used for TTS exports. Allowed: `approved`, `needs_review`, `reject`. |

## File Companions

For each manifest entry, the dataset bucket directory should also contain:

- `<id>.txt`: reviewed text, usually identical to `text`
- `<id>.wav` or `<id>.webm`: approved audio segment

Example:

```text
output/_sprachwerker/de-DE/bayern-callcenter-seed-mittel/
  0000000042.wav
  0000000042.txt
  _review_manifest.jsonl
```

## Compatibility Notes

- `text` is the canonical reviewed transcript and should be preferred by exporters.
- `raw_text` is optional and useful for audit or correction workflows.
- `dialect_label` is intentionally open-ended and should not be treated as a fixed enum.
- `style_label` currently remains only as a compatibility field for older manifests and future exporter flexibility.
- `emotion_label` follows a fixed Thorsten/Piper-like list in the current UI.
- `tts_suitability` should be treated as a hard filter for TTS exporters.
- Exporters should validate that required fields for their target pipeline are present instead of assuming every manifest line is valid for every training job.
- Older datasets may miss some workflow fields. Exporters should apply defaults or reject incomplete rows explicitly.
