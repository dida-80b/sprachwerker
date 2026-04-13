import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        print(f"[model-fetcher] missing setting: {name}", file=sys.stderr)
        raise SystemExit(2)
    return value


def ensure_file(repo_id: str, filename: str, target_dir: Path, token: str | None) -> None:
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[model-fetcher] exists: {target_path}")
        return

    print(f"[model-fetcher] downloading {filename} from {repo_id}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token or None,
    )
    print(f"[model-fetcher] ready: {target_path}")


def main() -> int:
    repo_id = env_value("VOXTRAL_HF_REPO", "bartowski/mistralai_Voxtral-Small-24B-2507-GGUF")
    model_filename = env_value("VOXTRAL_MODEL_FILENAME", "mistralai_Voxtral-Small-24B-2507-Q4_K_M.gguf")
    mmproj_filename = env_value("VOXTRAL_MMPROJ_FILENAME", "mmproj-mistralai_Voxtral-Small-24B-2507-f16.gguf")
    model_dir = Path(env_value("MODEL_DIR", "/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip() or None

    try:
        ensure_file(repo_id, model_filename, model_dir, token)
        ensure_file(repo_id, mmproj_filename, model_dir, token)
    except Exception as exc:
        print(f"[model-fetcher] download failed: {exc}", file=sys.stderr)
        if not token:
            print(
                "[model-fetcher] HF_TOKEN is optional for public repos, but required for gated repos.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
