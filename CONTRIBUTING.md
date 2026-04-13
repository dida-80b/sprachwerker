# Contributing to Sprachwerker Studio

Thanks for your interest in contributing.

## What is welcome

- Bug reports with reproduction steps
- Pull requests for bug fixes
- New language UI translations (see `UI_COPY` in `asr_api.py`)
- New emotion templates for additional languages (see `EMOTION_ACTOR_TEMPLATES`)
- Improvements to the dialog prompt quality
- Documentation improvements

## What to discuss first

Open an issue before starting work on:

- New major features
- Changes to the manifest schema (breaking change for existing datasets)
- Changes to the Docker stack architecture

## Development setup

```bash
cp .env.example .env
# edit .env as needed
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

The backend auto-reloads are not enabled by default. After changing `asr_api.py`:

```bash
docker cp asr_api.py sprachwerker:/app/asr_api.py
docker restart sprachwerker
```

After changing `templates/segment_review.html`:

```bash
docker cp templates/segment_review.html sprachwerker:/app/templates/segment_review.html
docker restart sprachwerker
```

## Adding a language

1. Add a new key to `UI_COPY` in `asr_api.py` — copy the `"de"` block as a starting point
2. Add labels for all `WORKFLOW_OPTIONS` values to `WORKFLOW_OPTION_LABELS`
3. Add templates for all `EMOTION_ACTOR_TEMPLATES` entries
4. Add fallback prompts to `FALLBACK_DIALOG_PROMPTS`
5. Test with `UI_DEFAULT_LANG=<your-lang>` in `.env`

## Code style

- Python: follow existing style, no formatter enforced
- Keep prompts and UI strings in the existing data structures — not inline in route handlers
- New API endpoints go after the existing endpoint block, not scattered through the file

## License

By contributing you agree that your contributions will be licensed under the AGPLv3.
