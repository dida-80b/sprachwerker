# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [Unreleased]

### Added
- Emotion-actor dialog mode: when a Kokoro emotion is selected, Piper actively triggers the emotion rather than asking a generic question — bad jokes for `happy`, provocations for `angry`, secrets for `whispering`, confusing questions for `question`, etc.
- Kokoro-compatible emotion label set: `neutral`, `happy`, `angry`, `surprised`, `sad`, `whispering`, `question` — replaces the old Piper/Thorsten list
- Delete button for dataset profiles (with confirmation dialog clarifying that only the profile is removed, not the recordings)
- English templates for all emotion-actor prompts
- Language-aware prompt selection in emotion-actor mode (`ui_lang` instead of hardcoded `"de"`)

### Changed
- Schema `emotion_label` enum updated to match Kokoro-compatible list
- `WORKFLOW_OPTION_LABELS` for both DE and EN synchronized with `WORKFLOW_OPTIONS`
- `style_hint` text updated to reflect Kokoro emotion list

### Fixed
- Path traversal vulnerability in `/segment-review/audio/{filename}` endpoint — filename is now sanitized and resolved path is verified to stay within the dataset directory
- Missing EN labels for `sad` and `question` in `WORKFLOW_OPTION_LABELS["en"]`
- Stale labels (`disgusted`, `drunk`, `sleepy`) removed from EN option labels

---

## [0.3.0] — Initial public release baseline

### Added
- Full dialog recording mode with Voxtral ASR + Piper TTS
- Automatic bootstrap of llama.cpp and Voxtral model via Docker services
- Bundled Piper service with `de_DE-thorsten-high` voice
- ROCm and CUDA docker-compose overlays
- Dataset profiles with emotion, dialect, task mode, and speaker profile metadata
- Duration-bucket routing (`kurz`, `mittel`, `lang`, `sehr-lang`)
- Duplicate and near-duplicate detection
- German and English UI
- JSON Lines manifest with machine-readable schema (`schemas/review_manifest.schema.json`)
- Translation mode: dialect/colloquial speech → Standard German via local LLM
