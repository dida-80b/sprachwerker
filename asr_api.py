"""
Sprachwerker backend.
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(APP_DIR / "output"))).resolve()
REVIEW_ROOT = OUTPUT_DIR / os.environ.get("REVIEW_SUBDIR", "_sprachwerker")
APP_VERSION = str(int(Path(__file__).stat().st_mtime))

APP_TITLE = os.environ.get("APP_TITLE", "Sprachwerker Studio")
DIALECT_NAME = os.environ.get("DIALECT_NAME", "Speech Collection")
SOURCE_TEXT_LABEL = os.environ.get("SOURCE_TEXT_LABEL", f"{DIALECT_NAME} / Raw transcript")
TARGET_TEXT_LABEL = os.environ.get("TARGET_TEXT_LABEL", "Approved text")
SOURCE_LANGUAGE_CODE = os.environ.get("SOURCE_LANGUAGE_CODE", "de-DE")
APP_REGION_GROUP = os.environ.get("APP_REGION_GROUP", "germany").strip().lower()
DATASET_PLACEHOLDER = os.environ.get("DATASET_PLACEHOLDER", "new-dataset-bulk")
TRANSLATION_ENABLED = os.environ.get("TRANSLATION_ENABLED", "1") != "0"
TTS_TARGET_HOURS = float(os.environ.get("TTS_TARGET_HOURS", "10"))
ASR_TARGET_HOURS = float(os.environ.get("ASR_TARGET_HOURS", "3"))
UI_DEFAULT_LANG = os.environ.get("UI_DEFAULT_LANG", "").lower()
DEFAULT_TASK_MODE = os.environ.get("DEFAULT_TASK_MODE", "both")
DEFAULT_AUDIO_DOMAIN = os.environ.get("DEFAULT_AUDIO_DOMAIN", "clean")
DEFAULT_TARGET_ENGINE = os.environ.get("DEFAULT_TARGET_ENGINE", "custom")
DEFAULT_SPEAKER_PROFILE = os.environ.get("DEFAULT_SPEAKER_PROFILE", "single_speaker")
DEFAULT_DIALECT_LABEL = os.environ.get("DEFAULT_DIALECT_LABEL", "")
DEFAULT_STYLE_LABEL = os.environ.get("DEFAULT_STYLE_LABEL", "neutral")
DEFAULT_EMOTION_LABEL = os.environ.get("DEFAULT_EMOTION_LABEL", "neutral")
DEFAULT_RECORDING_QUALITY = os.environ.get("DEFAULT_RECORDING_QUALITY", "clean")
DEFAULT_TTS_SUITABILITY = os.environ.get("DEFAULT_TTS_SUITABILITY", "approved")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5-tel:latest")
PIPER_URL = os.environ.get("PIPER_URL", "http://127.0.0.1:5150").rstrip("/")
PIPER_DEFAULT_VOICE = os.environ.get("PIPER_DEFAULT_VOICE", "de_DE-thorsten-high").strip()
VOXTRAL_TEXT_CLI_PATH = os.environ.get("VOXTRAL_TEXT_CLI_PATH", "/opt/llama-bin/llama-completion")
VOXTRAL_PROMPT_MAX_TOKENS = int(os.environ.get("VOXTRAL_PROMPT_MAX_TOKENS", "48"))
VOXTRAL_PROMPT_TIMEOUT_SECONDS = float(os.environ.get("VOXTRAL_PROMPT_TIMEOUT_SECONDS", "90"))
TRANSLATION_PROMPT_TEMPLATE = os.environ.get(
    "TRANSLATION_PROMPT_TEMPLATE",
    (
        "Uebertrage den folgenden Text aus dem deutschen Dialekt oder der Umgangssprache "
        "in gut lesbares Standarddeutsch. Erhalte die Bedeutung exakt. Gib nur die "
        "Zielversion zurueck, ohne Erklaerung.\n\n\"{text}\""
    ),
)

CLI_PATH = os.environ.get("VOXTRAL_CLI_PATH", "/opt/llama-bin/llama-mtmd-cli")
MODEL_PATH = os.environ.get("VOXTRAL_MODEL_PATH", "/models/mistralai_Voxtral-Small-24B-2507-Q4_K_M.gguf")
MMPROJ_PATH = os.environ.get("VOXTRAL_MMPROJ_PATH", "/models/mmproj-mistralai_Voxtral-Small-24B-2507-f16.gguf")
LD_LIBRARY_PATH = os.environ.get("VOXTRAL_LD_LIBRARY_PATH", "/opt/llama-bin:/opt/rocm/lib")
HIP_VISIBLE_DEVICES = os.environ.get("HIP_VISIBLE_DEVICES", "0")
VOXTRAL_PROMPTS = [
    prompt.strip()
    for prompt in os.environ.get(
        "VOXTRAL_PROMPTS",
        (
            "Du bist ein Speech-to-Text-System fuer deutsche Dialekte und Standarddeutsch. "
            "Schreibe exakt auf, was gesprochen wird. Dialekt, Fuellwoerter und Grammatikfehler "
            "bleiben erhalten. Nicht uebersetzen. Nicht verbessern. Nicht zusammenfassen. "
            "Gib nur den Transkripttext aus.||"
            "Verbatim transcription only. Preserve the spoken German dialect exactly. "
            "Do not normalize, translate, explain, or continue the sentence. "
            "Output only the spoken words from the audio.||"
            "Write the exact transcript of the audio in German, including dialect wording. "
            "If a word is unclear, keep the surrounding words and do not invent content. "
            "No explanation. No summary. No translation."
        ),
    ).split("||")
    if prompt.strip()
]
VOXTRAL_MAX_TOKENS = int(os.environ.get("VOXTRAL_MAX_TOKENS", "80"))
VOXTRAL_TIMEOUT_SECONDS = float(os.environ.get("VOXTRAL_TIMEOUT_SECONDS", "120"))
VOXTRAL_EVAL_MAX_TOKENS = int(os.environ.get("VOXTRAL_EVAL_MAX_TOKENS", "96"))
VOXTRAL_EVAL_TIMEOUT_SECONDS = float(os.environ.get("VOXTRAL_EVAL_TIMEOUT_SECONDS", "120"))
VOXTRAL_GPU_LAYERS = os.environ.get("VOXTRAL_GPU_LAYERS", "72")
VOXTRAL_CTX = os.environ.get("VOXTRAL_CTX", "2048")
VOXTRAL_BATCH = os.environ.get("VOXTRAL_BATCH", "512")
VOXTRAL_UBATCH = os.environ.get("VOXTRAL_UBATCH", "128")

DATASET_RE = re.compile(r"[^a-zA-Z0-9._-]+")
TRANSCRIPT_RE = re.compile(r"^[\w\u00c0-\u024f].*")
NOISE_PREFIXES = (
    "ggml_",
    "load_",
    "load:",
    "print_info:",
    "common_",
    "llama_",
    "sched_",
    "clip_",
    "init_audio:",
    "alloc_compute_meta:",
    "warmup:",
    "main:",
    "WARN:",
    "warning:",
    "system_info:",
    "mtmd_cli_context:",
)

app = FastAPI(title=APP_TITLE, version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
gpu_lock = asyncio.Lock()
logger = logging.getLogger("sprachwerker")

UI_COPY = {
    "de": {
        "lang_name": "Deutsch",
        "eyebrow": "Sprachwerker",
        "hero_copy": "Freisprechen, Satzende mit Leertaste markieren, danach segmentweise abhören, transkribieren, prüfen und als Datensatz für TTS oder ASR speichern.",
        "meta_focus": "Fokus",
        "meta_language": "Sprache",
        "meta_status": "Status",
        "menu_windows": "Fenster",
        "menu_language": "Sprache",
        "menu_help": "Hilfe",
        "menu_view": "Einblenden",
        "toggle_recording": "Aufnahme",
        "toggle_freerecord": "Freie Aufnahme",
        "toggle_checkup": "Checkup",
        "toggle_datasets": "Datensaetze",
        "toggle_settings": "Einstellungen",
        "freerecord_title": "Freie Aufnahme",
        "freerecord_hint": "Einfach drauflosreden — Pausen werden automatisch als Satzgrenzen erkannt.",
        "freerecord_start": "Aufnahme starten",
        "freerecord_stop": "Aufnahme stoppen",
        "freerecord_split": "Segmentieren",
        "freerecord_transcribe_all": "Alle transkribieren",
        "freerecord_save_all": "Alle speichern",
        "freerecord_no_segments": "Keine Segmente erkannt. Versuche laenger zu sprechen oder Pausen zu machen.",
        "freerecord_splitting": "Segmentiere...",
        "freerecord_transcribing": "Transkribiere {n} Segmente...",
        "freerecord_saved": "{n} Segmente gespeichert.",
        "freerecord_min_silence": "Mindest-Pause (ms)",
        "freerecord_discard": "Verwerfen",
        "open_settings": "Einstellungen",
        "open_help_window": "Hilfe-Fenster",
        "popup_blocked": "Popup blockiert. Bitte Browser-Popups fuer diese Seite erlauben.",
        "settings_workflow": "Datensatz anlegen",
        "settings_group_base": "Basis",
        "settings_group_language": "Sprache / Dialekt",
        "settings_group_style": "Emotion",
        "settings_group_base_help": "Lege hier fest, ob der Datensatz fuer TTS, ASR oder beides gedacht ist. Mehrere Sprecher sind aktuell nur fuer reine ASR-Datensaetze sinnvoll und erlaubt.",
        "tts_multispeaker_warning": "Mehrere Sprecher sind aktuell nur fuer reine ASR-Datensaetze erlaubt. Fuer TTS oder Mischsaetze wird automatisch auf einen Sprecher begrenzt.",
        "settings_group_language_help": "Hier geht es im Studio nur noch um Sprache bzw. Dialekt. Der Audiotyp bleibt intern auf clean und wird spaeter bei Imports separat gesetzt.",
        "settings_group_style_help": "Standard-Emotion fuer neue Aufnahmen. Kann jederzeit geaendert werden — kein neues Profil noetig. Piper reagiert dann automatisch passend.",
        "dataset_select_label": "Aktiver Datensatz",
        "dataset_select_hint": "Datensatzprofil einmal anlegen, danach hier nur noch auswaehlen und aufnehmen.",
        "dataset_create_name": "Datensatzname",
        "dataset_create_name_hint": "z. B. xyz-niederbairisch-drunken. Nach dem Anlegen bleibt das Profil fix.",
        "dataset_create_button": "Datensatz anlegen",
        "dataset_create_note": "Name und Grundeinstellungen werden einmalig festgelegt. Emotion und Dialekt kann man danach jederzeit aendern.",
        "dataset_profiles_title": "Vorhandene Datensaetze",
        "dataset_profile_locked": "Profil fixiert",
        "dataset_profile_missing": "Bitte zuerst einen Datensatz anlegen.",
        "dataset_profile_exists": "Datensatz existiert bereits.",
        "dataset_profile_created": "Datensatz angelegt.",
        "recording_workspace": "Aufnahme",
        "recording_workspace_note": "Start spielt den Prompt automatisch ab. Danach startet Leertaste die Aufnahme und die naechste Leertaste stoppt sie wieder.",
        "dialog_title": "Dialog mit KI",
        "dialog_note": "Voxtral erzeugt den Prompttext, Piper spricht ihn. Aufgenommen wird nur dein Mikrofon.",
        "dialog_prompt_type": "Prompt-Art",
        "dialog_prompt_length": "Prompt-Länge",
        "dialog_tts_voice": "Piper-Stimme",
        "dialog_prompt_generate": "Prompt sprechen",
        "dialog_prompt_next": "Nächster Prompt",
        "dialog_prompt_repeat": "Nochmal",
        "dialog_prompt_empty": "Noch kein Prompt. Leertaste startet den ersten Prompt.",
        "dialog_prompt_current": "Prompt",
        "dialog_prompt_status_ready": "Bereit für Dialog-Aufnahme.",
        "dialog_prompt_status_loading": "Prompt wird erzeugt...",
        "dialog_prompt_status_speaking": "Prompt wird abgespielt...",
        "dialog_prompt_status_countdown": "Aufnahme startet in",
        "dialog_prompt_status_recording": "Aufnahme aktiv. Leertaste setzt Satzende.",
        "dialog_prompt_status_processing": "Aufnahme wird verarbeitet...",
        "dialog_prompt_status_audio_fallback": "Piper nicht erreichbar. Prompt steht oben und die Aufnahme kann trotzdem per Leertaste gestartet werden.",
        "dialog_prompt_status_voice_unavailable": "Piper-Stimme nicht erreichbar. Aufnahme funktioniert trotzdem mit sichtbarem Prompt.",
        "dialog_prompt_status_stopped": "Prompt gestoppt.",
        "dialog_prompt_status_error": "Dialog-Prompt fehlgeschlagen.",
        "dialog_prompt_status_backend_voxtral": "Voxtral aktiv.",
        "dialog_prompt_status_backend_fallback": "VOXTRAL FEHLER. Ersatzprompt aktiv.",
        "dialog_mode_idle": "Bereit",
        "dialog_mode_loading": "Prompt holen",
        "dialog_mode_speaking": "Prompt spricht",
        "dialog_mode_countdown": "Countdown",
        "dialog_mode_starting": "Startet",
        "dialog_mode_recording": "Aufnahme",
        "dialog_mode_processing": "Verarbeite",
        "dialog_countdown_idle": "Warte",
        "dialog_prompt_type_everyday": "Alltag",
        "dialog_prompt_type_question": "Frage",
        "dialog_prompt_type_story": "Erzählen",
        "dialog_prompt_type_emotion": "Emotion",
        "dialog_prompt_type_random": "Zufall",
        "dialog_prompt_length_short": "Kurz",
        "dialog_prompt_length_medium": "Mittel",
        "dialog_voice_loading": "Piper-Stimmen werden geladen...",
        "dialog_voice_error": "Piper nicht erreichbar.",
        "dialog_voice_none": "Keine Piper-Stimme gefunden.",
        "quality_green": "TTS + ASR",
        "quality_yellow": "Nur ASR",
        "quality_red": "Verwerfen",
        "quality_pending": "Noch nicht bewertet",
        "checkup_title": "Checkup",
        "tts_goal": "TTS-Ziel",
        "tts_hint": "Für TTS zählen vor allem saubere Daten eines einzelnen Sprechers mit stabiler Aussprache.",
        "asr_goal": "ASR-Ziel",
        "asr_hint": "Für ASR sind echte Zielbedingungen wichtig: Telefon, Alltagssprache, Sprecherwechsel und Dialekte.",
        "duplicate_title": "Doppelte Sätze",
        "similar_title": "Sehr ähnliche Sätze",
        "dataset_base_label": "Basis-Datensatzname",
        "dataset_bucket_hint": "Beim Speichern wird automatisch nach Satzlänge aufgeteilt.",
        "workflow_title": "Workflow-Metadaten",
        "task_mode": "Modus",
        "speaker_profile": "Sprecherprofil",
        "dialect_label": "Dialekt / Akzent / Variante",
        "dialect_placeholder": "z. B. Bairisch, Swiss German, Tyrolean English",
        "emotion_label": "Emotion",
        "style_hint": "Kokoro-kompatibel: neutral, happy, angry, surprised, sad, whispering, question",
        "record_start": "Start",
        "record_pause": "Pause",
        "record_stop_play": "Stop/Play",
        "record_stop": "Stop",
        "record_ready": "Bereit. Leertaste startet die Aufnahme. Nochmals Leertaste stoppt sie.",
        "record_running": "Aufnahme läuft. Bei jedem Satzende Leertaste drücken.",
        "record_paused": "Pausiert.",
        "record_waiting_next": "Prompt fertig. Leertaste startet die Aufnahme.",
        "record_processing": "Audio wird in Segmente zerlegt...",
        "record_idle": "Idle",
        "recording": "Recording",
        "processing": "Processing",
        "ready": "Ready",
        "space_hint_recording": "Leertaste während der Aufnahme: Segmentgrenze.",
        "space_hint_idle": "Nach dem Prompt startet Leertaste die Aufnahme. Die naechste Leertaste stoppt sie wieder.",
        "asr_all": "Alle transkribieren",
        "save_approved": "Alle speichern",
        "datasets_title": "Datensätze",
        "review_focus_title": "Prüffokus",
        "close": "Schließen",
        "help_title": "Hilfe",
        "show": "Einblenden",
        "hide": "Ausblenden",
        "help_tts_title": "TTS",
        "help_tts_body_1": "Für eine einzelne Stimme sind saubere Aufnahmen eines einzelnen Sprechers am wertvollsten.",
        "help_tts_body_2": "Gut nutzbar sind vor allem kurze bis mittlere Segmente mit stabiler Artikulation.",
        "help_asr_title": "ASR",
        "help_asr_body_1": "Für ASR ist zielnahes Audio entscheidend: echte Telefonaudios, verschiedene Sprecher und natürliche Alltagssprache.",
        "help_asr_body_2": "Wenige Stunden zielnaher Daten können schon viel bringen.",
        "help_text_title": "Textregel",
        "help_text_body_1": "Nicht unnötig glätten. Wenn Satzlogik, Dialekt oder Akzent relevant sind, darf das sichtbar bleiben.",
        "help_bucket_title": "Auto-Buckets",
        "help_bucket_body_1": "Gespeicherte Segmente werden automatisch nach Audiolänge in kurz, mittel, lang und sehr-lang aufgeteilt.",
        "help_duplicates_title": "Dubletten & Ähnlichkeit",
        "help_duplicates_body_1": "Exakte Duplikate sind für TTS und ASR meist wenig hilfreich. Sehr ähnliche Satzschablonen erhöhen zwar die Menge, aber nicht zwingend die Abdeckung.",
        "segment_none": "Noch keine Segmente. Aufnahme starten, sprechen, Satzende mit Leertaste markieren.",
        "dataset_none": "Noch keine gespeicherten Datensätze.",
        "dataset_empty": "Dieser Datensatz ist leer.",
        "duplicate_none": "Keine exakten Duplikate gefunden.",
        "similar_none": "Keine auffällig ähnlichen Sätze gefunden.",
        "segment_label": "Satz",
        "saved": "gespeichert",
        "approved": "freigegeben",
        "open": "offen",
        "play": "Play",
        "transcribe": "ASR",
        "translate": "Standard",
        "approve": "Freigeben",
        "approved_action": "Freigegeben",
        "delete": "Löschen",
        "listen_label": "Segment anhören und freigeben",
        "listen_hint": "Vor dem Freigeben kurz gegenhören.",
        "approved_text_label": "Freigegebener Text",
        "approved_text_placeholder": "Hier Zieltext prüfen und korrigieren...",
        "save": "Speichern",
        "review": "Bearbeiten",
        "keep": "Behalten",
        "unkeep": "Keep lösen",
        "kept": "Bewusst behalten",
        "open_in_dataset": "Im Datensatz öffnen",
        "migrate": "Migrieren",
        "entries": "Einträge",
        "variants": "Varianten",
        "duplicate_x": "Dublette",
        "similar_near": "Nähe",
        "dataset_name_required": "Bitte Datensatznamen setzen.",
        "nothing_to_save": "Keine neuen freigegebenen Segmente zum Speichern.",
        "saving": "Segmente werden gespeichert...",
        "save_failed": "Speichern fehlgeschlagen.",
        "saved_segments": "Segmente gespeichert",
        "transcribe_running": "ASR läuft",
        "transcribe_done": "ASR fertig.",
        "transcribe_error": "ASR Fehler.",
        "translate_running": "Standarddeutsch läuft...",
        "translate_done": "Standarddeutsch fertig.",
        "translate_error": "Übersetzung fehlgeschlagen.",
        "legacy_confirm": "Legacy-Datensatz jetzt in Längen-Buckets migrieren?",
        "delete_dataset_confirm": "Datensatz wirklich löschen?",
        "delete_item_confirm": "Eintrag wirklich löschen?",
        "keep_confirm": "Alle Einträge als bewusst behalten markieren?",
        "keep_failed": "Keep-Markierung fehlgeschlagen.",
        "item_updated": "Eintrag aktualisiert.",
        "segment_count_summary": "{segments} Segmente, {approved} freigegeben, {saved} gespeichert",
        "segments_recognized": "{count} Segmente erkannt.",
        "no_segments_recognized": "Keine Segmente erkannt.",
        "focus_label": "Dataset Studio fuer TTS, ASR und Sprachvarianten",
        "asr_note": "ASR wird spaeter noch nach Telefon, Clean, Sprecherzahl und Variante getrennt.",
    },
    "en": {
        "lang_name": "English",
        "eyebrow": "Sprachwerker",
        "hero_copy": "Speak freely, mark sentence boundaries with the space bar, then review, transcribe and store segments as TTS or ASR datasets.",
        "meta_focus": "Focus",
        "meta_language": "Language",
        "meta_status": "Status",
        "menu_windows": "Windows",
        "menu_language": "Language",
        "menu_help": "Help",
        "menu_view": "Show",
        "toggle_recording": "Recording",
        "toggle_freerecord": "Free Recording",
        "toggle_checkup": "Checkup",
        "toggle_datasets": "Datasets",
        "toggle_settings": "Settings",
        "freerecord_title": "Free Recording",
        "freerecord_hint": "Just talk freely — pauses are automatically detected as sentence boundaries.",
        "freerecord_start": "Start recording",
        "freerecord_stop": "Stop recording",
        "freerecord_split": "Segment",
        "freerecord_transcribe_all": "Transcribe all",
        "freerecord_save_all": "Save all",
        "freerecord_no_segments": "No segments detected. Try speaking longer or making clear pauses.",
        "freerecord_splitting": "Segmenting...",
        "freerecord_transcribing": "Transcribing {n} segments...",
        "freerecord_saved": "{n} segments saved.",
        "freerecord_min_silence": "Min. pause (ms)",
        "freerecord_discard": "Discard",
        "open_settings": "Settings",
        "open_help_window": "Help window",
        "popup_blocked": "Popup blocked. Please allow popups for this page.",
        "settings_workflow": "Create dataset",
        "settings_group_base": "Basics",
        "settings_group_language": "Language / dialect",
        "settings_group_style": "Emotion",
        "settings_group_base_help": "Define whether the dataset is meant for TTS, ASR or both. Multiple speakers are currently only allowed for pure ASR datasets.",
        "tts_multispeaker_warning": "Multiple speakers are currently only allowed for pure ASR datasets. TTS and mixed datasets are forced to a single speaker.",
        "settings_group_language_help": "Inside the studio this is now only about language and dialect. Audio domain stays internally on clean and can be set later by import workflows.",
        "settings_group_style_help": "Default emotion for new recordings. Can be changed at any time — no new profile needed. Piper reacts automatically.",
        "dataset_select_label": "Active dataset",
        "dataset_select_hint": "Create the dataset profile once, then only select it here and record.",
        "dataset_create_name": "Dataset name",
        "dataset_create_name_hint": "for example xyz-lower-bavarian-drunken. After creation the profile stays fixed.",
        "dataset_create_button": "Create dataset",
        "dataset_create_note": "Name and base settings are set once. Emotion and dialect can be changed at any time afterwards.",
        "dataset_profiles_title": "Existing datasets",
        "dataset_profile_locked": "Profile locked",
        "dataset_profile_missing": "Please create a dataset first.",
        "dataset_profile_exists": "Dataset already exists.",
        "dataset_profile_created": "Dataset created.",
        "recording_workspace": "Recording",
        "recording_workspace_note": "Start plays the prompt automatically. Then space starts recording and the next space stops it again.",
        "dialog_title": "Dialog with AI",
        "dialog_note": "Voxtral generates the prompt text, Piper speaks it. Only your microphone is recorded.",
        "dialog_prompt_type": "Prompt type",
        "dialog_prompt_length": "Prompt length",
        "dialog_tts_voice": "Piper voice",
        "dialog_prompt_generate": "Speak prompt",
        "dialog_prompt_next": "Next prompt",
        "dialog_prompt_repeat": "Repeat",
        "dialog_prompt_empty": "No prompt yet. Press space to start the first prompt.",
        "dialog_prompt_current": "Prompt",
        "dialog_prompt_status_ready": "Ready for dialog recording.",
        "dialog_prompt_status_loading": "Generating prompt...",
        "dialog_prompt_status_speaking": "Playing prompt...",
        "dialog_prompt_status_countdown": "Recording starts in",
        "dialog_prompt_status_recording": "Recording live. Press space at each sentence boundary.",
        "dialog_prompt_status_processing": "Processing recording...",
        "dialog_prompt_status_audio_fallback": "Piper is unavailable. The prompt stays visible and recording can still be started with space.",
        "dialog_prompt_status_voice_unavailable": "Piper voice is unavailable. Recording still works with the visible prompt.",
        "dialog_prompt_status_stopped": "Prompt stopped.",
        "dialog_prompt_status_error": "Dialog prompt failed.",
        "dialog_prompt_status_backend_voxtral": "Voxtral active.",
        "dialog_prompt_status_backend_fallback": "VOXTRAL ERROR. Fallback prompt active.",
        "dialog_mode_idle": "Ready",
        "dialog_mode_loading": "Loading",
        "dialog_mode_speaking": "Speaking",
        "dialog_mode_countdown": "Countdown",
        "dialog_mode_starting": "Starting",
        "dialog_mode_recording": "Recording",
        "dialog_mode_processing": "Processing",
        "dialog_countdown_idle": "Waiting",
        "dialog_prompt_type_everyday": "Everyday",
        "dialog_prompt_type_question": "Question",
        "dialog_prompt_type_story": "Story",
        "dialog_prompt_type_emotion": "Emotion",
        "dialog_prompt_type_random": "Random",
        "dialog_prompt_length_short": "Short",
        "dialog_prompt_length_medium": "Medium",
        "dialog_voice_loading": "Loading Piper voices...",
        "dialog_voice_error": "Piper is not reachable.",
        "dialog_voice_none": "No Piper voice found.",
        "quality_green": "TTS + ASR",
        "quality_yellow": "ASR only",
        "quality_red": "Discard",
        "quality_pending": "Not rated yet",
        "checkup_title": "Checkup",
        "tts_goal": "TTS target",
        "tts_hint": "For TTS, clean single-speaker recordings with stable pronunciation matter most.",
        "asr_goal": "ASR target",
        "asr_hint": "For ASR, target conditions matter most: telephony, spontaneous speech, speaker variation and dialects.",
        "duplicate_title": "Duplicate sentences",
        "similar_title": "Very similar sentences",
        "dataset_base_label": "Base dataset name",
        "dataset_bucket_hint": "Saved segments are bucketed automatically by duration.",
        "workflow_title": "Workflow metadata",
        "task_mode": "Mode",
        "speaker_profile": "Speaker profile",
        "dialect_label": "Dialect / accent / variety",
        "dialect_placeholder": "for example Bavarian, Swiss German, Tyrolean English",
        "emotion_label": "Emotion",
        "style_hint": "Kokoro-compatible: neutral, happy, angry, surprised, sad, whispering, question",
        "record_start": "Start",
        "record_pause": "Pause",
        "record_stop_play": "Stop/Play",
        "record_stop": "Stop",
        "record_ready": "Ready. Space starts recording. Press space again to stop.",
        "record_running": "Recording. Press space at every sentence boundary.",
        "record_paused": "Paused.",
        "record_waiting_next": "Prompt finished. Space starts recording.",
        "record_processing": "Splitting audio into segments...",
        "record_idle": "Idle",
        "recording": "Recording",
        "processing": "Processing",
        "ready": "Ready",
        "space_hint_recording": "Space during recording: add segment boundary.",
        "space_hint_idle": "After the prompt, press space to start recording. Press space again to stop it.",
        "asr_all": "Transcribe all",
        "save_approved": "Save all",
        "datasets_title": "Datasets",
        "review_focus_title": "Review focus",
        "close": "Close",
        "help_title": "Help",
        "show": "Show",
        "hide": "Hide",
        "help_tts_title": "TTS",
        "help_tts_body_1": "For a single voice, clean recordings from one speaker are most valuable.",
        "help_tts_body_2": "Short to mid-length segments with stable articulation work best.",
        "help_asr_title": "ASR",
        "help_asr_body_1": "For ASR, target-like audio wins: real phone audio, multiple speakers and natural everyday speech.",
        "help_asr_body_2": "A few hours of high-match data can already move the model a lot.",
        "help_text_title": "Text policy",
        "help_text_body_1": "Do not over-normalize. Keep dialect, accent and sentence logic visible when they matter.",
        "help_bucket_title": "Auto buckets",
        "help_bucket_body_1": "Saved segments are split automatically into short, medium, long and very-long duration buckets.",
        "help_duplicates_title": "Duplicates & similarity",
        "help_duplicates_body_1": "Exact duplicates rarely help TTS or ASR. Very similar templates increase volume, but not necessarily coverage.",
        "segment_none": "No segments yet. Start recording, speak, and mark sentence boundaries with the space bar.",
        "dataset_none": "No saved datasets yet.",
        "dataset_empty": "This dataset is empty.",
        "duplicate_none": "No exact duplicates found.",
        "similar_none": "No suspiciously similar sentence groups found.",
        "segment_label": "Sentence",
        "saved": "saved",
        "approved": "approved",
        "open": "open",
        "play": "Play",
        "transcribe": "ASR",
        "translate": "Normalize",
        "approve": "Approve",
        "approved_action": "Approved",
        "delete": "Delete",
        "listen_label": "Listen and approve",
        "listen_hint": "Quickly listen back before approving.",
        "approved_text_label": "Approved text",
        "approved_text_placeholder": "Review and correct the target text here...",
        "save": "Save",
        "review": "Edit",
        "keep": "Keep",
        "unkeep": "Unset keep",
        "kept": "Kept intentionally",
        "open_in_dataset": "Open in dataset",
        "migrate": "Migrate",
        "entries": "entries",
        "variants": "variants",
        "duplicate_x": "duplicate",
        "similar_near": "similarity",
        "dataset_name_required": "Please set a dataset name.",
        "nothing_to_save": "No new approved segments to save.",
        "saving": "Saving segments...",
        "save_failed": "Saving failed.",
        "saved_segments": "segments saved",
        "transcribe_running": "ASR running",
        "transcribe_done": "ASR done.",
        "transcribe_error": "ASR error.",
        "translate_running": "Normalization running...",
        "translate_done": "Normalization done.",
        "translate_error": "Normalization failed.",
        "legacy_confirm": "Migrate this legacy dataset into duration buckets now?",
        "delete_dataset_confirm": "Delete this dataset?",
        "delete_item_confirm": "Delete this item?",
        "keep_confirm": "Mark all selected items as intentionally kept?",
        "keep_failed": "Failed to update keep flag.",
        "item_updated": "Item updated.",
        "segment_count_summary": "{segments} segments, {approved} approved, {saved} saved",
        "segments_recognized": "{count} segments detected.",
        "no_segments_recognized": "No segments detected.",
        "focus_label": "Dataset studio for TTS, ASR and speech varieties",
        "asr_note": "ASR should later be split further by telephony, clean audio, speaker count and language variety.",
    },
}

WORKFLOW_OPTIONS = {
    "task_mode": ["tts", "asr", "both"],
    "audio_domain": ["clean", "telephone", "noisy"],
    "target_engine": ["piper", "kokoro", "whisper", "faster-whisper", "custom"],
    "speaker_profile": ["single_speaker", "multi_speaker"],
    "emotion_label": ["neutral", "happy", "angry", "surprised", "sad", "whispering", "question"],
    "recording_quality": ["clean", "usable", "noisy"],
    "tts_suitability": ["approved", "needs_review", "reject"],
}

WORKFLOW_OPTION_LABELS = {
    "de": {
        "tts": "TTS",
        "asr": "ASR",
        "both": "TTS + ASR",
        "clean": "Clean",
        "telephone": "Telefon",
        "noisy": "Rauschig",
        "piper": "Piper",
        "kokoro": "Kokoro",
        "whisper": "Whisper",
        "faster-whisper": "Faster-Whisper",
        "custom": "Custom",
        "single_speaker": "Ein Sprecher",
        "multi_speaker": "Mehrere Sprecher",
        "neutral": "Neutral",
        "happy": "Fröhlich / Lachen",
        "angry": "Wütend",
        "surprised": "Überrascht",
        "sad": "Traurig",
        "whispering": "Flüsternd",
        "question": "Fragend / Nachdenklich",
        "usable": "Brauchbar",
        "noisy": "Rauschig",
        "approved": "Freigegeben",
        "needs_review": "Prüfen",
        "reject": "Verwerfen",
    },
    "en": {
        "tts": "TTS",
        "asr": "ASR",
        "both": "TTS + ASR",
        "clean": "Clean",
        "telephone": "Telephone",
        "noisy": "Noisy",
        "piper": "Piper",
        "kokoro": "Kokoro",
        "whisper": "Whisper",
        "faster-whisper": "Faster-Whisper",
        "custom": "Custom",
        "single_speaker": "Single speaker",
        "multi_speaker": "Multiple speakers",
        "neutral": "Neutral",
        "happy": "Happy / Laughing",
        "angry": "Angry",
        "surprised": "Surprised",
        "sad": "Sad",
        "whispering": "Whispering",
        "question": "Questioning / Thoughtful",
        "usable": "Usable",
        "approved": "Approved",
        "needs_review": "Needs review",
        "reject": "Reject",
    },
}

PROMPT_TYPE_GUIDANCE = {
    "everyday": {
        "de": "Alltagsfrage oder kurze alltagsnahe Aufforderung.",
        "en": "Everyday question or short everyday prompt.",
    },
    "question": {
        "de": "Direkte Frage, leicht frei beantwortbar.",
        "en": "Direct question that is easy to answer freely.",
    },
    "story": {
        "de": "Kleine Erzaehlaufforderung fuer 1 bis 3 Saetze.",
        "en": "Small storytelling prompt for one to three sentences.",
    },
    "emotion": {
        "de": "Kurze Aufforderung, die zur Datensatz-Emotion passt.",
        "en": "Short prompt that matches the dataset emotion.",
    },
}

# Emotion-spezifische Prompt-Vorlagen: Piper agiert, User reagiert.
# Ollama generiert Text den Piper SPRICHT — nicht eine Frage an den User.
EMOTION_ACTOR_TEMPLATES = {
    "happy": {
        "en": (
            "Generate exactly one short bad joke or funny observation in English.\n"
            "Rules:\n"
            "- Piper tells the joke itself — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Can be cheesy or absurd — should make someone laugh.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the joke. Now:\n"
            "Joke:"
        ),
        "de": (
            "Du erzeugst genau einen schlechten deutschen Witz oder eine lustige Beobachtung.\n"
            "Regeln:\n"
            "- Piper erzählt den Witz selbst — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Darf flach, cheesy oder absurd sein — soll zum Lachen reizen.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur den Witz aus. Jetzt:\n"
            "Witz:"
        ),
        "fallback": [
            "Warum können Geister so schlecht lügen? Weil man durch sie hindurchsieht.",
            "Ich habe mir ein Buch über Paranoia gekauft. Jemand hat alle Seiten umgeknickt.",
            "Was ist orange und klingt wie ein Papagei? Eine Karotte.",
            "Warum hat der Mathematiker Angst vor negativen Zahlen? Er schreckt vor nichts zurück.",
            "Was sagt ein Elektriker wenn er sich verliebt? Es hat mich voll erwischt.",
        ],
    },
    "angry": {
        "en": (
            "Generate exactly one provocative or mildly annoying statement in English.\n"
            "Rules:\n"
            "- Piper says something wrong or stupid — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Should cause mild outrage — cheeky but not offensive.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the statement. Now:\n"
            "Statement:"
        ),
        "de": (
            "Du erzeugst genau eine provokante, leicht nervende Aussage auf Deutsch.\n"
            "Regeln:\n"
            "- Piper sagt etwas Dummes oder Falsches — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Soll eine leichte Empörung auslösen — nicht beleidigend, aber frech.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur die Aussage aus. Jetzt:\n"
            "Aussage:"
        ),
        "fallback": [
            "Eigentlich ist Ananas auf Pizza eine kulinarische Meisterleistung.",
            "Wer kein Frühstück isst ist selber schuld wenn er müde ist.",
            "Bayerisch ist eigentlich kein richtiger Dialekt, das ist nur schlechtes Deutsch.",
            "Filterkaffee schmeckt sowieso besser als Espresso.",
            "Wer zu spät kommt den bestraft das Leben, das war immer schon so.",
        ],
    },
    "whispering": {
        "en": (
            "Generate exactly one whispered secret or confidential piece of information in English.\n"
            "Rules:\n"
            "- Piper reveals something confidential — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Should make the listener feel they must respond quietly.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the secret. Now:\n"
            "Secret:"
        ),
        "de": (
            "Du erzeugst genau ein geflüstertes Geheimnis oder eine vertrauliche Information auf Deutsch.\n"
            "Regeln:\n"
            "- Piper verrät etwas Vertrauliches — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Soll das Gefühl erzeugen dass man leise antworten muss.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur das Geheimnis aus. Jetzt:\n"
            "Geheimnis:"
        ),
        "fallback": [
            "Psst — ich glaube der Chef weiß schon dass wir früher gegangen sind.",
            "Sag das niemandem aber der neue Kollege hat sich glaube ich vertan.",
            "Ich hab gehört dass das Meeting morgen ausfällt, aber offiziell weiß das noch keiner.",
            "Zwischen uns: das Rezept hat eine geheime Zutat die sie nie verraten würden.",
            "Pass auf, ich glaube da drüben hört jemand zu.",
        ],
    },
    "question": {
        "en": (
            "Generate exactly one confusing or thought-provoking question in English.\n"
            "Rules:\n"
            "- Piper asks a real question — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Should trigger a hesitant, searching, questioning response.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the question. Now:\n"
            "Question:"
        ),
        "de": (
            "Du erzeugst genau eine verwirrende oder nachdenklich machende Frage auf Deutsch.\n"
            "Regeln:\n"
            "- Piper stellt eine echte Frage — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Soll ein zögerndes, suchendes, fragendes Antwortverhalten auslösen.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur die Frage aus. Jetzt:\n"
            "Frage:"
        ),
        "fallback": [
            "Wenn du eine Farbe wärst die es noch nicht gibt, wie würde sie klingen?",
            "Was würdest du tun wenn du wüsstest dass niemand zuschaut?",
            "Gibt es etwas das du weißt aber nie aussprechen würdest?",
            "Wann hast du zuletzt etwas getan ohne zu wissen warum?",
            "Wenn du deinem jüngeren Ich einen einzigen Satz sagen dürftest, welcher wäre das?",
        ],
    },
    "surprised": {
        "en": (
            "Generate exactly one surprising or unexpected announcement in English.\n"
            "Rules:\n"
            "- Piper announces something unexpected — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Should trigger genuine surprise or astonishment.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the statement. Now:\n"
            "Statement:"
        ),
        "de": (
            "Du erzeugst genau eine überraschende oder unerwartete Aussage auf Deutsch.\n"
            "Regeln:\n"
            "- Piper verkündet etwas Unerwartetes — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Soll echtes Erstaunen oder Überraschung auslösen.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur die Aussage aus. Jetzt:\n"
            "Aussage:"
        ),
        "fallback": [
            "Übrigens — das Meeting von heute war eigentlich für morgen geplant.",
            "Du hast gerade aus Versehen den Hauptgewinn gewonnen.",
            "Der Kollege von gestern war eigentlich der neue Chef.",
            "Das System hat gerade alle deine Dateien als perfekt bewertet.",
            "Anscheinend war das die letzte Aufgabe — du bist fertig.",
        ],
    },
    "sad": {
        "en": (
            "Generate exactly one melancholic or quietly reflective statement in English.\n"
            "Rules:\n"
            "- Piper says something still and wistful — no instruction to the user.\n"
            "- Maximum {max_words} words.\n"
            "- Should prompt a calm, slightly subdued response.\n"
            "- Do not repeat: {history_text}\n\n"
            "Output only the statement. Now:\n"
            "Statement:"
        ),
        "de": (
            "Du erzeugst genau eine melancholische oder nachdenklich stimmende Aussage auf Deutsch.\n"
            "Regeln:\n"
            "- Piper sagt etwas Stilles, Wehmütiges — kein Auftrag an den User.\n"
            "- Maximal {max_words} Wörter.\n"
            "- Soll eine ruhige, leicht traurige Antwort auslösen.\n"
            "- Nicht wiederholen: {history_text}\n\n"
            "Gib nur die Aussage aus. Jetzt:\n"
            "Aussage:"
        ),
        "fallback": [
            "Manchmal vermisst man Dinge bevor man weiß dass sie weg sind.",
            "Es gibt Tage die einfach schwerer wiegen als andere.",
            "Manche Gespräche hätte man früher führen sollen.",
            "Einige Momente wiederholen sich nie — das merkt man meist zu spät.",
            "Es ist seltsam still wenn jemand fehlt der immer da war.",
        ],
    },
}

PROMPT_TYPE_GUIDANCE["random"] = {
    "de": "Natuerliche Mischung aus Frage, Alltag und Erzaehlung.",
    "en": "Natural mix of question, everyday prompt and storytelling.",
}

FALLBACK_DIALOG_PROMPTS = {
    "de": {
        "everyday": [
            "Was hast du heute gern gegessen?",
            "Wie klingt ein guter Morgen für dich?",
            "Was machst du gern am Wochenende?",
            "Worauf freust du dich gerade?",
        ],
        "question": [
            "Was entspannt dich am schnellsten?",
            "Welche Arbeit machst du wirklich gern?",
            "Was war heute dein bester Moment?",
            "Worüber könntest du spontan erzählen?",
        ],
        "story": [
            "Erzähl kurz von einem ruhigen Abend.",
            "Erzähl von einer kleinen Panne.",
            "Erzähl von einem schönen Weg.",
            "Erzähl von einem kurzen Besuch.",
        ],
        "emotion": [
            "Erzähl von etwas, das dich überrascht hat.",
            "Erzähl von etwas, das dich gefreut hat.",
            "Erzähl von einem anstrengenden Moment.",
            "Erzähl von etwas, das dich genervt hat.",
        ],
        "random": [
            "Wie war dein letzter freie Moment?",
            "Was würdest du heute noch gern machen?",
            "Erzähl kurz von einer kleinen Gewohnheit.",
            "Was macht einen Tag für dich gut?",
        ],
    },
    "en": {
        "everyday": [
            "What did you enjoy eating today?",
            "What makes a morning feel good to you?",
            "What do you like doing on weekends?",
            "What are you looking forward to?",
        ],
        "question": [
            "What helps you relax quickly?",
            "What kind of work do you enjoy?",
            "What was your best moment today?",
            "What could you talk about right away?",
        ],
        "story": [
            "Tell a short story about a quiet evening.",
            "Tell a short story about a small mistake.",
            "Tell a short story about a nice walk.",
            "Tell a short story about a short visit.",
        ],
        "emotion": [
            "Tell me about something that surprised you.",
            "Tell me about something that made you happy.",
            "Tell me about a stressful moment.",
            "Tell me about something annoying.",
        ],
        "random": [
            "How was your last free moment?",
            "What would you still like to do today?",
            "Tell me about a small habit you have.",
            "What makes a day feel good to you?",
        ],
    },
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_ui_lang(value: str | None) -> str:
    configured = (value or UI_DEFAULT_LANG or "").lower()
    if configured in UI_COPY:
        return configured
    lang = "de" if APP_REGION_GROUP == "germany" else "en"
    return lang if lang in UI_COPY else "de"


def ui_config(ui_lang: str | None = None) -> dict[str, Any]:
    lang = normalize_ui_lang(ui_lang)
    strings = UI_COPY[lang]
    return {
        "appTitle": APP_TITLE,
        "dialectName": DIALECT_NAME,
        "sourceTextLabel": SOURCE_TEXT_LABEL,
        "targetTextLabel": TARGET_TEXT_LABEL,
        "sourceLanguageCode": SOURCE_LANGUAGE_CODE,
        "regionGroup": APP_REGION_GROUP,
        "datasetPlaceholder": DATASET_PLACEHOLDER,
        "translationEnabled": TRANSLATION_ENABLED,
        "ttsTargetHours": TTS_TARGET_HOURS,
        "asrTargetHours": ASR_TARGET_HOURS,
        "uiLanguage": lang,
        "strings": strings,
        "workflowDefaults": {
            "taskMode": DEFAULT_TASK_MODE,
            "audioDomain": DEFAULT_AUDIO_DOMAIN,
            "targetEngine": DEFAULT_TARGET_ENGINE,
            "speakerProfile": DEFAULT_SPEAKER_PROFILE,
            "dialectLabel": DEFAULT_DIALECT_LABEL,
            "emotionLabel": DEFAULT_EMOTION_LABEL,
            "recordingQuality": DEFAULT_RECORDING_QUALITY,
            "ttsSuitability": DEFAULT_TTS_SUITABILITY,
        },
        "workflowOptions": WORKFLOW_OPTIONS,
        "workflowOptionLabels": WORKFLOW_OPTION_LABELS.get(lang, {}),
        "bucketTargetsSeconds": {
            "kurz": 45 * 60,
            "mittel": 90 * 60,
            "lang": 90 * 60,
            "sehr-lang": 45 * 60,
        },
        "appVersion": APP_VERSION,
    }


def sanitize_dataset_name(name: str | None) -> str:
    if not name:
        return ""
    return DATASET_RE.sub("-", name.strip()).strip("-.")[:80]


def sanitize_item_id(item_id: str | None) -> str:
    if not item_id:
        return ""
    return re.sub(r"[^0-9]", "", item_id)[:10]


def clean_generated_prompt(text: str) -> str:
    cleaned = text.replace("\r", " ").strip()
    for marker in (
        "Neuer Prompt:", "New prompt:",
        "Witz:", "Joke:",
        "Aussage:", "Statement:",
        "Geheimnis:", "Secret:",
        "Frage:", "Question:",
        "Prompt:",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()
            break
    cleaned = re.sub(r"\[end of text\].*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(User|Assistant|System)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    # Anführungszeichen entfernen die das Modell manchmal um den Output setzt
    cleaned = cleaned.strip()
    cleaned = re.sub(r'^[„"\']|[""\']+$', "", cleaned).strip()
    return cleaned


def extract_completion_text(output: str) -> str:
    cleaned_output = output
    for marker in (
        "Neuer Prompt:", "New prompt:",
        "Witz:", "Joke:",
        "Aussage:", "Statement:",
        "Geheimnis:", "Secret:",
        "Frage:", "Question:",
        "Prompt:",
    ):
        if marker in cleaned_output:
            cleaned_output = cleaned_output.split(marker)[-1]
            break

    lines = []
    for raw_line in cleaned_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(NOISE_PREFIXES):
            continue
        if line.startswith(("build:", "main:", "load_", "print_info:", "system_info:", "common_", "llama_", "sampler ", "generate:", "sched_", "ggml_", "load:", "Device ", "repeat_last_n", "top_k =", "mirostat =", "dry_multiplier =", "top_p =", "sampler chain:")):
            continue
        if "tokens per second" in line:
            continue
        lines.append(line)

    text = clean_generated_prompt(" ".join(lines))
    if not text:
        raise ValueError("No prompt text found in llama-completion output")
    return text


def extract_rating_reason(output: str) -> dict[str, str]:
    rating_match = re.search(r"RATING\s*:\s*(green|yellow|red)", output, flags=re.IGNORECASE)
    reason_match = re.search(r"REASON\s*:\s*(.+)", output, flags=re.IGNORECASE)
    if not rating_match:
        raise ValueError("No rating found in model output")
    rating = rating_match.group(1).strip().lower()
    reason = reason_match.group(1).strip() if reason_match else "Audio pruefen."
    return {"rating": rating, "reason": reason}


def dataset_profile_for_prompt(language: str, dataset: str) -> dict[str, Any]:
    profiles = load_dataset_profiles(language)
    return profiles.get(dataset, {})


def fallback_dialog_prompt(language: str, prompt_type: str, history: list[str], dataset: str = "", emotion_override: str = "") -> str:
    ui_lang = "de" if language.lower().startswith("de") else "en"
    # Emotion-Actor-Fallback: feste Texte aus EMOTION_ACTOR_TEMPLATES
    if dataset:
        profile = dataset_profile_for_prompt(language, dataset)
        stored_emotion = (profile.get("emotionLabel") or DEFAULT_EMOTION_LABEL).strip()
    else:
        stored_emotion = DEFAULT_EMOTION_LABEL
    emotion = emotion_override.strip() if emotion_override.strip() else stored_emotion
    # Auto-promote to emotion mode when non-neutral emotion is active
    if emotion and emotion != "neutral" and prompt_type == "random":
        prompt_type = "emotion"
    if prompt_type == "emotion":
        if emotion in EMOTION_ACTOR_TEMPLATES:
            options = list(EMOTION_ACTOR_TEMPLATES[emotion].get("fallback", []))
            if options:
                seen = {e.strip().lower() for e in history if isinstance(e, str) and e.strip()}
                fresh = [o for o in options if o.strip().lower() not in seen]
                return random.choice(fresh or options)
    prompt_type = prompt_type if prompt_type in FALLBACK_DIALOG_PROMPTS[ui_lang] else "random"
    options = list(FALLBACK_DIALOG_PROMPTS[ui_lang][prompt_type])
    seen = {entry.strip().lower() for entry in history if isinstance(entry, str) and entry.strip()}
    fresh = [item for item in options if item.strip().lower() not in seen]
    pool = fresh or options
    return random.choice(pool)


def build_dialog_prompt_request(
    language: str,
    dataset: str,
    prompt_type: str,
    prompt_length: str,
    history: list[str],
    emotion_override: str = "",
) -> str:
    profile = dataset_profile_for_prompt(language, dataset)
    ui_lang = "de" if language.lower().startswith("de") else "en"
    prompt_length = prompt_length if prompt_length in {"short", "medium"} else "short"
    max_words = 12 if prompt_length == "short" else 20
    dialect = (profile.get("dialectLabel") or "").strip() or ("Deutsch" if ui_lang == "de" else "German")
    emotion = emotion_override.strip() if emotion_override.strip() else (profile.get("emotionLabel") or DEFAULT_EMOTION_LABEL).strip()
    # Auto-promote to emotion mode when a non-neutral emotion is set
    if prompt_type not in PROMPT_TYPE_GUIDANCE:
        prompt_type = "random"
    if emotion and emotion != "neutral" and prompt_type == "random":
        prompt_type = "emotion"
    task_mode = profile.get("taskMode") or DEFAULT_TASK_MODE
    history_text = "\n".join(f"- {entry}" for entry in history[-6:] if entry.strip()) or "- (keine)"
    # Emotion-Actor-Modus: Piper agiert, User reagiert
    if prompt_type == "emotion" and emotion in EMOTION_ACTOR_TEMPLATES:
        actor = EMOTION_ACTOR_TEMPLATES[emotion]
        template = actor.get(ui_lang) or actor.get("de", "")
        if template:
            history_flat = ", ".join(f'"{e}"' for e in history[-6:] if e.strip()) or ("(none)" if ui_lang == "en" else "(keine)")
            return template.format(max_words=max_words, history_text=history_flat)

    guidance = PROMPT_TYPE_GUIDANCE[prompt_type][ui_lang]
    if ui_lang == "de":
        return (
            "Du erzeugst genau einen natuerlich gesprochenen Prompt fuer eine Sprachaufnahme.\n"
            "Regeln:\n"
            "- Gib genau einen einzigen Promptsatz oder eine kurze Frage aus.\n"
            "- Keine Liste. Keine Erklaerung. Keine Einleitung. Keine Rollenbezeichnung.\n"
            "- Maximal {max_words} Woerter.\n"
            "- Gut frei beantwortbar mit 1 bis 3 gesprochenen Saetzen.\n"
            "- Kein Bezug auf Uhrzeit, Datum, Wetter oder den aktuellen Ort.\n"
            "- Nicht wiederholen, was schon benutzt wurde.\n"
            "- Wenn Emotion nicht neutral ist, darf die Situation leicht dazu passen, aber bleibe alltagstauglich.\n\n"
            "Datensatz:\n"
            f"- Sprache: {language}\n"
            f"- Modus: {task_mode}\n"
            f"- Dialekt/Variante: {dialect}\n"
            f"- Emotion: {emotion}\n"
            f"- Prompt-Art: {guidance}\n\n"
            "Bisherige Prompts:\n"
            f"{history_text}\n\n"
            "Jetzt genau einen neuen Prompt ausgeben.\n"
            "Neuer Prompt:"
        ).format(max_words=max_words)
    return (
        "Generate exactly one natural spoken prompt for a speech recording.\n"
        "Rules:\n"
        "- Output exactly one short prompt or one short question.\n"
        "- No list. No explanation. No label. No preface.\n"
        f"- Maximum {max_words} words.\n"
        "- It should be easy to answer freely in one to three spoken sentences.\n"
        "- Avoid time, date, weather or current-location questions.\n"
        "- Do not repeat previous prompts.\n"
        "- If the target emotion is not neutral, the situation may lightly match it.\n\n"
        "Dataset:\n"
        f"- Language: {language}\n"
        f"- Mode: {task_mode}\n"
        f"- Dialect/variety: {dialect}\n"
        f"- Emotion: {emotion}\n"
        f"- Prompt type: {guidance}\n\n"
        "Previous prompts:\n"
        f"{history_text}\n\n"
        "Output exactly one new prompt.\n"
        "New prompt:"
    )


def build_quality_eval_prompt(language: str, dataset: str, transcript: str, duration_seconds: float) -> str:
    profile = dataset_profile_for_prompt(language, dataset)
    task_mode = profile.get("taskMode") or DEFAULT_TASK_MODE
    dialect = (profile.get("dialectLabel") or DEFAULT_DIALECT_LABEL).strip()
    emotion = (profile.get("emotionLabel") or DEFAULT_EMOTION_LABEL).strip()
    return (
        "Bewerte das Audiosegment fuer Datensatznutzung. Sei streng fuer TTS.\n"
        "Audio und gesprochener Fluss sind wichtiger als bloesser Text.\n"
        "Achte auf:\n"
        "- lange Denkpausen\n"
        "- hoerbares Zoegern oder langes Ueberlegen\n"
        "- Fuelllaute wie aeh, aehm, hm\n"
        "- Satzabbrueche oder Neustarts\n"
        "- unklaren oder unsauberen Abschluss\n"
        "- allgemeine TTS-Tauglichkeit\n\n"
        "Bewertung:\n"
        '- green = gut fuer TTS und ASR\n'
        '- yellow = fuer ASR okay, fuer TTS eher ungeeignet\n'
        '- red = fuer den Datensatz verwerfen\n\n'
        "Gib nur genau diese zwei Zeilen zurueck:\n"
        "RATING: green|yellow|red\n"
        "REASON: kurzer deutscher Grund\n\n"
        f"Kontext:\n- Sprache: {language}\n- Datensatz: {dataset}\n- Modus: {task_mode}\n"
        f"- Dialekt: {dialect or 'unbekannt'}\n- Emotion: {emotion or 'neutral'}\n"
        f"- Dauer in Sekunden: {duration_seconds:.2f}\n"
        f"- Vorlaeufiges Transkript: {transcript or '(leer)'}"
    )


def filler_only_rating(transcript: str) -> dict[str, str] | None:
    cleaned = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", " ", transcript.lower())
    words = [word for word in cleaned.split() if word]
    if not words:
        return None

    filler_words = {
        "ja",
        "jo",
        "joa",
        "joaeh",
        "aeh",
        "aehm",
        "äh",
        "ähm",
        "hm",
        "hmm",
        "mhm",
        "mmh",
        "oha",
        "naja",
        "okay",
        "ok",
        "gut",
        "also",
    }
    filler_count = sum(1 for word in words if word in filler_words)
    if len(words) <= 4 and filler_count == len(words):
        return {"rating": "yellow", "reason": "Nur Fuelllaute oder sehr kurze Rueckmeldung."}
    if len(words) <= 5 and filler_count / len(words) >= 0.6:
        return {"rating": "yellow", "reason": "Stark von Fuelllauten oder Kurzantworten gepraegt."}
    return None


def dataset_dir(language: str, dataset: str) -> Path:
    return REVIEW_ROOT / language / dataset


def dataset_profiles_path(language: str) -> Path:
    return REVIEW_ROOT / language / "_dataset_profiles.json"


def load_dataset_profiles(language: str) -> dict[str, dict[str, Any]]:
    path = dataset_profiles_path(language)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    profiles = raw if isinstance(raw, dict) else {}
    return {sanitize_dataset_name(name): value for name, value in profiles.items() if sanitize_dataset_name(name)}


def save_dataset_profiles(language: str, profiles: dict[str, dict[str, Any]]) -> None:
    path = dataset_profiles_path(language)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {name: profiles[name] for name in sorted(profiles)}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classify_duration_bucket(duration_seconds: float) -> str:
    if duration_seconds < 3.5:
        return "kurz"
    if duration_seconds < 6.5:
        return "mittel"
    if duration_seconds < 10.0:
        return "lang"
    if duration_seconds < 16.0:
        return "sehr-lang"
    return "zu-lang"


def normalize_similarity_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\säöüÄÖÜß]", " ", text.lower())
    return " ".join(cleaned.split())


def similarity_score(text_a: str, text_b: str) -> tuple[float, float]:
    norm_a = normalize_similarity_text(text_a)
    norm_b = normalize_similarity_text(text_b)
    if not norm_a or not norm_b:
        return 0.0, 0.0

    seq_score = SequenceMatcher(None, norm_a, norm_b).ratio()
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return seq_score, 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return seq_score, jaccard


def get_next_segment_id(group_dir: Path) -> int:
    max_id = 0
    for text_path in group_dir.glob("*.txt"):
        try:
            max_id = max(max_id, int(text_path.stem))
        except ValueError:
            continue
    return max_id + 1


def list_datasets(language: str) -> list[dict[str, Any]]:
    language_dir = REVIEW_ROOT / language
    if not language_dir.exists():
        return []

    datasets = []
    for group_dir in sorted((p for p in language_dir.iterdir() if p.is_dir()), reverse=True):
        items = []
        manifest_lookup: dict[str, dict[str, Any]] = {}
        manifest_path = group_dir / "_review_manifest.jsonl"
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                manifest_lookup[str(entry.get("id", ""))] = entry

        for text_path in sorted(group_dir.glob("*.txt")):
            audio_path = None
            for candidate in (group_dir / f"{text_path.stem}.wav", group_dir / f"{text_path.stem}.webm"):
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is None:
                continue
            entry = manifest_lookup.get(text_path.stem, {})
            duration_seconds = float(entry.get("duration_seconds") or 0.0)
            if not duration_seconds and entry.get("start") is not None and entry.get("end") is not None:
                try:
                    duration_seconds = max(0.0, float(entry["end"]) - float(entry["start"]))
                except (TypeError, ValueError):
                    duration_seconds = 0.0
            items.append(
                {
                    "id": text_path.stem,
                    "text": text_path.read_text(encoding="utf-8").strip(),
                    "rawText": entry.get("raw_text", ""),
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "durationSeconds": duration_seconds,
                    "bucket": entry.get("bucket") or classify_duration_bucket(duration_seconds),
                    "keepMarked": bool(entry.get("keep_marked", False)),
                    "audioUrl": f"/segment-review/audio/{language}/{group_dir.name}/{audio_path.name}",
                    "audioPath": str(audio_path.relative_to(OUTPUT_DIR)),
                    "sizeBytes": audio_path.stat().st_size,
                    "baseDataset": entry.get("base_dataset", ""),
                    "taskMode": entry.get("task_mode", "both"),
                    "audioDomain": entry.get("audio_domain", "clean"),
                    "targetEngine": entry.get("target_engine", "custom"),
                    "speakerProfile": entry.get("speaker_profile", "single_speaker"),
                    "dialectLabel": entry.get("dialect_label", ""),
                    "emotionLabel": entry.get("emotion_label", "neutral"),
                    "recordingQuality": entry.get("recording_quality", "clean"),
                    "ttsSuitability": entry.get("tts_suitability", "approved"),
                }
            )

        total_duration = sum(float(item.get("durationSeconds") or 0.0) for item in items)
        datasets.append(
            {
                "name": group_dir.name,
                "language": language,
                "count": len(items),
                "durationSeconds": total_duration,
                "items": items,
            }
        )
    return datasets


def list_dataset_profiles(language: str) -> list[dict[str, Any]]:
    profiles = load_dataset_profiles(language)
    datasets = list_datasets(language)
    counts_by_base: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        for item in dataset.get("items", []):
            base_dataset = sanitize_dataset_name(item.get("baseDataset") or item.get("base_dataset") or "")
            if not base_dataset:
                base_dataset = sanitize_dataset_name(dataset.get("name", "").rsplit("-", 1)[0])
            if not base_dataset:
                continue
            summary = counts_by_base.setdefault(base_dataset, {"count": 0, "durationSeconds": 0.0})
            summary["count"] += 1
            summary["durationSeconds"] += float(item.get("durationSeconds") or 0.0)

    result = []
    for name in sorted(profiles):
        result.append(
            {
                "name": name,
                "workflow": profiles[name],
                "count": counts_by_base.get(name, {}).get("count", 0),
                "durationSeconds": counts_by_base.get(name, {}).get("durationSeconds", 0.0),
            }
        )
    return result


def build_checkup(language: str, ui_lang: str | None = None) -> dict[str, Any]:
    config = ui_config(ui_lang)
    datasets = list_datasets(language)
    bucket_totals = {name: 0.0 for name in ("kurz", "mittel", "lang", "sehr-lang", "zu-lang")}
    total_duration = 0.0
    total_items = 0
    all_items: list[dict[str, Any]] = []

    for dataset in datasets:
        for item in dataset["items"]:
            duration_seconds = float(item.get("durationSeconds") or 0.0)
            bucket = item.get("bucket") or classify_duration_bucket(duration_seconds)
            bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + duration_seconds
            total_duration += duration_seconds
            total_items += 1
            all_items.append({"dataset": dataset["name"], **item})

    bucket_targets = config["bucketTargetsSeconds"]
    bucket_progress = []
    for key, target_seconds in bucket_targets.items():
        current = bucket_totals.get(key, 0.0)
        bucket_progress.append(
            {
                "bucket": key,
                "seconds": current,
                "hours": current / 3600.0,
                "targetSeconds": target_seconds,
                "targetHours": target_seconds / 3600.0,
                "percent": min(100.0, (current / target_seconds * 100.0) if target_seconds else 0.0),
            }
        )

    duplicates_map: dict[str, list[dict[str, Any]]] = {}
    normalized_items = []
    for item in all_items:
        norm = " ".join((item.get("text") or "").strip().lower().split())
        if not norm:
            continue
        duplicates_map.setdefault(norm, []).append(item)
        normalized_items.append((norm, item))

    duplicate_groups = []
    for norm_text, items in duplicates_map.items():
        if len(items) > 1:
            visible_items = [entry for entry in items if not entry.get("keepMarked")]
            if len(visible_items) <= 1:
                continue
            duplicate_groups.append(
                {
                    "text": visible_items[0]["text"],
                    "count": len(visible_items),
                    "items": [{"dataset": entry["dataset"], "id": entry["id"], "text": entry["text"]} for entry in visible_items],
                }
            )

    similar_pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, (text_a_norm, item_a) in enumerate(normalized_items):
        for text_b_norm, item_b in normalized_items[index + 1:]:
            if item_a.get("keepMarked") or item_b.get("keepMarked"):
                continue
            pair_key = tuple(sorted((f"{item_a['dataset']}:{item_a['id']}", f"{item_b['dataset']}:{item_b['id']}")))
            if pair_key in seen_pairs:
                continue
            if text_a_norm == text_b_norm:
                continue
            words_a = text_a_norm.split()
            words_b = text_b_norm.split()
            if min(len(words_a), len(words_b)) < 5:
                continue
            if abs(len(words_a) - len(words_b)) > 4:
                continue

            seq_score, jaccard = similarity_score(item_a["text"], item_b["text"])
            combined_score = (seq_score * 0.6) + (jaccard * 0.4)
            if seq_score >= 0.90 and jaccard >= 0.60 and combined_score >= 0.82:
                similar_pairs.append(
                    {
                        "score": round(combined_score, 2),
                        "sequenceScore": round(seq_score, 2),
                        "jaccardScore": round(jaccard, 2),
                        "a": {"dataset": item_a["dataset"], "id": item_a["id"], "text": item_a["text"]},
                        "b": {"dataset": item_b["dataset"], "id": item_b["id"], "text": item_b["text"]},
                    }
                )
                seen_pairs.add(pair_key)

    similar_pairs = sorted(similar_pairs, key=lambda entry: entry["score"], reverse=True)[:12]

    adjacency: dict[str, set[str]] = {}
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    item_lookup: dict[str, dict[str, Any]] = {}
    for item in all_items:
        item_key = f"{item['dataset']}:{item['id']}"
        item_lookup[item_key] = item
        adjacency.setdefault(item_key, set())
    for pair in similar_pairs:
        key_a = f"{pair['a']['dataset']}:{pair['a']['id']}"
        key_b = f"{pair['b']['dataset']}:{pair['b']['id']}"
        adjacency.setdefault(key_a, set()).add(key_b)
        adjacency.setdefault(key_b, set()).add(key_a)
        pair_lookup[tuple(sorted((key_a, key_b)))] = pair

    similar_groups = []
    visited_nodes: set[str] = set()
    for node in adjacency:
        if node in visited_nodes or not adjacency[node]:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in visited_nodes:
                continue
            visited_nodes.add(current)
            component.add(current)
            stack.extend(adjacency.get(current, set()) - visited_nodes)

        if len(component) < 2:
            continue

        component_pairs = []
        for a in component:
            for b in component:
                if a >= b:
                    continue
                pair = pair_lookup.get(tuple(sorted((a, b))))
                if pair:
                    component_pairs.append(pair)

        if not component_pairs:
            continue

        scores = [pair["score"] for pair in component_pairs]
        group_items = []
        for item_key in sorted(component):
            item = item_lookup[item_key]
            group_items.append(
                {
                    "dataset": item["dataset"],
                    "id": item["id"],
                    "text": item["text"],
                    "keepMarked": item.get("keepMarked", False),
                }
            )

        similar_groups.append(
            {
                "count": len(group_items),
                "score": round(max(scores), 2),
                "minScore": round(min(scores), 2),
                "items": group_items,
                "representativeText": group_items[0]["text"],
            }
        )

    similar_groups = sorted(similar_groups, key=lambda entry: (entry["score"], entry["count"]), reverse=True)

    return {
        "totalItems": total_items,
        "totalHours": total_duration / 3600.0,
        "datasets": len(datasets),
        "bucketProgress": bucket_progress,
        "tts": {
            "targetHours": TTS_TARGET_HOURS,
            "hours": total_duration / 3600.0,
            "percent": min(100.0, (total_duration / 3600.0) / TTS_TARGET_HOURS * 100.0 if TTS_TARGET_HOURS else 0.0),
        },
        "asr": {
            "targetHours": ASR_TARGET_HOURS,
            "hours": total_duration / 3600.0,
            "percent": min(100.0, (total_duration / 3600.0) / ASR_TARGET_HOURS * 100.0 if ASR_TARGET_HOURS else 0.0),
            "note": config["strings"]["asr_note"],
        },
        "duplicates": duplicate_groups,
        "similar": similar_groups,
    }


def migrate_legacy_dataset(language: str, dataset_name: str) -> dict[str, Any]:
    safe_name = sanitize_dataset_name(dataset_name)
    if not safe_name:
        return {"success": False, "error": "Invalid dataset"}
    if safe_name.endswith(("-kurz", "-mittel", "-lang", "-sehr-lang", "-zu-lang")):
        return {"success": False, "error": "Dataset is already bucketed"}

    source_dir = dataset_dir(language, safe_name)
    if not source_dir.exists():
        return {"success": False, "error": "Dataset not found"}

    manifest_path = source_dir / "_review_manifest.jsonl"
    manifest_entries: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            manifest_entries[str(entry.get("id", ""))] = entry

    migrated: dict[str, int] = {}
    text_files = sorted(source_dir.glob("*.txt"))
    for text_path in text_files:
        stem = text_path.stem
        audio_path = None
        audio_ext = None
        for ext in ("wav", "webm"):
            candidate = source_dir / f"{stem}.{ext}"
            if candidate.exists():
                audio_path = candidate
                audio_ext = ext
                break
        if audio_path is None:
            continue

        entry = manifest_entries.get(stem, {})
        duration_seconds = 0.0
        try:
            if entry.get("duration_seconds") is not None:
                duration_seconds = float(entry["duration_seconds"])
            elif entry.get("start") is not None and entry.get("end") is not None:
                duration_seconds = max(0.0, float(entry["end"]) - float(entry["start"]))
        except (TypeError, ValueError):
            duration_seconds = 0.0

        bucket = entry.get("bucket") or classify_duration_bucket(duration_seconds)
        target_dataset = sanitize_dataset_name(f"{safe_name}-{bucket}")
        target_dir = dataset_dir(language, target_dataset)
        target_dir.mkdir(parents=True, exist_ok=True)
        next_id = get_next_segment_id(target_dir)
        new_stem = f"{next_id:010d}"

        shutil.move(str(text_path), str(target_dir / f"{new_stem}.txt"))
        shutil.move(str(audio_path), str(target_dir / f"{new_stem}.{audio_ext}"))

        migrated_entry = {
            "id": new_stem,
            "text": text_path.read_text(encoding="utf-8") if False else None,
        }
        migrated_metadata = {
            "id": new_stem,
            "text": (target_dir / f"{new_stem}.txt").read_text(encoding="utf-8"),
            "raw_text": entry.get("raw_text", ""),
            "start": entry.get("start"),
            "end": entry.get("end"),
            "duration_seconds": duration_seconds,
            "bucket": bucket,
            "base_dataset": safe_name,
            "task_mode": entry.get("task_mode", "both"),
            "audio_domain": entry.get("audio_domain", "clean"),
            "target_engine": entry.get("target_engine", "custom"),
            "speaker_profile": entry.get("speaker_profile", "single_speaker"),
            "dialect_label": entry.get("dialect_label", ""),
            "style_label": entry.get("style_label", "neutral"),
            "emotion_label": entry.get("emotion_label", "neutral"),
            "recording_quality": entry.get("recording_quality", "clean"),
            "tts_suitability": entry.get("tts_suitability", "approved"),
        }
        with (target_dir / "_review_manifest.jsonl").open("a", encoding="utf-8") as target_manifest:
            target_manifest.write(json.dumps(migrated_metadata, ensure_ascii=False) + "\n")
        migrated[target_dataset] = migrated.get(target_dataset, 0) + 1

    if manifest_path.exists():
        manifest_path.unlink()

    remaining_files = list(source_dir.iterdir()) if source_dir.exists() else []
    if not remaining_files:
        source_dir.rmdir()

    return {"success": True, "migrated": migrated, "dataset": safe_name}


def normalize_audio(audio_bytes: bytes) -> tuple[bytes, float]:
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as src:
            src.write(audio_bytes)
            src_path = Path(src.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
            dst_path = Path(dst.name)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(src_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(dst_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")
            data, samplerate = sf.read(str(dst_path), always_2d=True)
        finally:
            src_path.unlink(missing_ok=True)
            dst_path.unlink(missing_ok=True)
    duration_seconds = float(data.shape[0]) / float(samplerate)

    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)

    if samplerate != 16000:
        new_length = int(data.shape[0] * 16000 / samplerate)
        x_old = np.linspace(0, 1, data.shape[0], endpoint=False)
        x_new = np.linspace(0, 1, new_length, endpoint=False)
        resampled = np.empty((new_length, 1), dtype=np.float32)
        resampled[:, 0] = np.interp(x_new, x_old, data[:, 0])
        data = resampled

    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, data, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue(), duration_seconds


def extract_transcript(output: str) -> str:
    audio_done = re.search(r"audio decoded \(batch \d+/\d+\) in \d+ ms\s*", output)
    if audio_done:
        output = output[audio_done.end():]

    perf_start = output.find("llama_perf_context_print:")
    if perf_start >= 0:
        output = output[:perf_start]

    lines = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(NOISE_PREFIXES):
            continue
        if line.startswith(("build:", "load_backend:", "Device ", "Using ", "This GDB supports", "For normal use cases", "https://")):
            continue
        if line.startswith("[") and "LWP" in line:
            continue
        if set(line) == {"."}:
            continue
        if "tokens per second" in line:
            continue
        if not TRANSCRIPT_RE.match(line):
            continue
        lines.append(line)

    if not lines:
        raise ValueError("No transcript found in llama-mtmd-cli output")

    return " ".join(lines).strip()


async def run_audio_cli_raw(audio_path: Path, prompt: str, max_tokens: int, timeout_seconds: float) -> str:
    cmd = [
        CLI_PATH,
        "-m",
        MODEL_PATH,
        "--mmproj",
        MMPROJ_PATH,
        "--audio",
        str(audio_path),
        "-p",
        prompt,
        "-ngl",
        VOXTRAL_GPU_LAYERS,
        "-c",
        VOXTRAL_CTX,
        "-b",
        VOXTRAL_BATCH,
        "-ub",
        VOXTRAL_UBATCH,
        "-fa",
        "off",
        "--no-warmup",
        "-n",
        str(max_tokens),
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = HIP_VISIBLE_DEVICES
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise TimeoutError("Audio model timeout") from exc

    output = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(output.strip() or f"llama-mtmd-cli failed with code {proc.returncode}")
    return output


async def run_cli_with_prompt(audio_path: Path, prompt: str, max_tokens: int) -> str:
    output = await run_audio_cli_raw(audio_path, prompt, max_tokens, VOXTRAL_TIMEOUT_SECONDS)
    return extract_transcript(output)


async def run_text_completion(prompt: str, max_tokens: int) -> str:
    cmd = [
        VOXTRAL_TEXT_CLI_PATH,
        "-m",
        MODEL_PATH,
        "-ngl",
        VOXTRAL_GPU_LAYERS,
        "-c",
        VOXTRAL_CTX,
        "-b",
        VOXTRAL_BATCH,
        "-ub",
        VOXTRAL_UBATCH,
        "--no-warmup",
        "-n",
        str(max_tokens),
        "-no-cnv",
        "-p",
        prompt,
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = HIP_VISIBLE_DEVICES
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=VOXTRAL_PROMPT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise TimeoutError("Prompt generation timeout") from exc

    output = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(output.strip() or f"llama-completion failed with code {proc.returncode}")
    return extract_completion_text(output)


def choose_max_tokens(duration_seconds: float) -> int:
    if duration_seconds <= 2.5:
        return min(40, VOXTRAL_MAX_TOKENS)
    if duration_seconds <= 5.0:
        return min(56, VOXTRAL_MAX_TOKENS)
    if duration_seconds <= 9.0:
        return min(72, VOXTRAL_MAX_TOKENS)
    return VOXTRAL_MAX_TOKENS


def looks_bad_transcript(text: str, duration_seconds: float) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True

    bad_prefixes = (
        "sure,",
        "please provide",
        "i'm unable",
        "ich bin",
        "du bist",
        "transcribe this",
        "write the exact",
    )
    if lowered.startswith(bad_prefixes):
        return True

    words = re.findall(r"\S+", text)
    if len(words) > max(8, int(duration_seconds * 5.0)):
        return True
    return False


@app.on_event("startup")
async def startup_event() -> None:
    ensure_dirs()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, ui_lang: str = UI_DEFAULT_LANG) -> HTMLResponse:
    config = ui_config(ui_lang)
    response = templates.TemplateResponse(
        request=request,
        name="segment_review.html",
        context={"request": request, "ui_config": config, "ui_strings": config["strings"]},
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/segment-review", response_class=HTMLResponse)
async def segment_review(request: Request, ui_lang: str = UI_DEFAULT_LANG) -> HTMLResponse:
    return await root(request, ui_lang=ui_lang)


@app.get("/api/config")
async def api_config(ui_lang: str = UI_DEFAULT_LANG) -> dict[str, Any]:
    return ui_config(ui_lang)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    audio_bytes = await audio.read()

    try:
        wav_bytes, duration_seconds = normalize_audio(audio_bytes)
    except Exception as exc:
        logger.warning("transcribe audio normalization failed: %s", exc)
        return JSONResponse(status_code=400, content={"success": False, "error": f"Audio processing error: {exc}"})

    async with gpu_lock:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)

        try:
            max_tokens = choose_max_tokens(duration_seconds)
            best_text = ""
            errors = []
            for prompt in VOXTRAL_PROMPTS:
                try:
                    candidate = await run_cli_with_prompt(tmp_path, prompt, max_tokens)
                    if not looks_bad_transcript(candidate, duration_seconds):
                        best_text = candidate
                        break
                    if not best_text:
                        best_text = candidate
                except Exception as exc:
                    errors.append(str(exc))
            if not best_text:
                logger.warning("transcribe failed to produce candidate: %s", "; ".join(errors) if errors else "unknown error")
                raise RuntimeError("; ".join(errors) if errors else "No transcription candidate produced")
            return JSONResponse(content={"success": True, "text": best_text, "model": "voxtral-small-24b-q4_k_m"})
        except TimeoutError as exc:
            logger.warning("transcribe timeout: %s", exc)
            return JSONResponse(status_code=504, content={"success": False, "error": str(exc)})
        except Exception as exc:
            logger.warning("transcribe failed: %s", exc)
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)


@app.post("/evaluate")
async def evaluate_segment(
    audio: UploadFile = File(...),
    dataset: str = Form(...),
    language: str = Form(SOURCE_LANGUAGE_CODE),
    transcript: str = Form(""),
) -> JSONResponse:
    dataset = sanitize_dataset_name(dataset)
    transcript = transcript.strip()
    if not dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset required"})

    filler_rating = filler_only_rating(transcript)
    if filler_rating is not None:
        return JSONResponse(
            content={
                "success": True,
                "rating": filler_rating["rating"],
                "reason": filler_rating["reason"],
                "durationSeconds": 0.0,
            }
        )

    audio_bytes = await audio.read()
    try:
        wav_bytes, duration_seconds = normalize_audio(audio_bytes)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Audio processing error: {exc}"})

    eval_prompt = build_quality_eval_prompt(language, dataset, transcript, duration_seconds)
    async with gpu_lock:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)

        try:
            output = await run_audio_cli_raw(tmp_path, eval_prompt, VOXTRAL_EVAL_MAX_TOKENS, VOXTRAL_EVAL_TIMEOUT_SECONDS)
            data = extract_rating_reason(output)
            rating = str(data.get("rating") or "yellow").strip().lower()
            if rating not in {"green", "yellow", "red"}:
                rating = "yellow"
            reason = str(data.get("reason") or "").strip() or "Audio pruefen."
            return JSONResponse(
                content={
                    "success": True,
                    "rating": rating,
                    "reason": reason,
                    "durationSeconds": duration_seconds,
                }
            )
        except TimeoutError as exc:
            return JSONResponse(status_code=504, content={"success": False, "error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)


@app.post("/translate")
async def translate(request: Request) -> JSONResponse:
    payload = await request.json()
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse(content={"success": True, "translation": ""})
    if not TRANSLATION_ENABLED:
        return JSONResponse(content={"success": True, "translation": text})

    prompt = TRANSLATION_PROMPT_TEMPLATE.format(text=text)
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                OLLAMA_ENDPOINT,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            result = response.json()
            translation = (result.get("response") or "").strip() or text
            return JSONResponse(content={"success": True, "translation": translation})
    except Exception as exc:
        return JSONResponse(content={"success": True, "translation": text, "warning": str(exc)})


@app.get("/dialog/voices")
async def dialog_voices() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{PIPER_URL}/voices")
            response.raise_for_status()
            result = response.json()
            return JSONResponse(
                content={
                    "success": True,
                    "voices": result.get("voices") or [],
                    "defaultVoice": PIPER_DEFAULT_VOICE,
                }
            )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": str(exc), "voices": [], "defaultVoice": PIPER_DEFAULT_VOICE},
        )


@app.post("/dialog/prompt")
async def dialog_prompt(request: Request) -> JSONResponse:
    payload = await request.json()
    language = payload.get("language") or SOURCE_LANGUAGE_CODE
    dataset = sanitize_dataset_name(payload.get("dataset"))
    prompt_type = str(payload.get("promptType") or "random").strip().lower()
    prompt_length = str(payload.get("promptLength") or "short").strip().lower()
    emotion_override = str(payload.get("emotion") or "").strip().lower()
    history = payload.get("history") or []
    if not dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset required"})

    history_list = history if isinstance(history, list) else []
    prompt_request = build_dialog_prompt_request(language, dataset, prompt_type, prompt_length, history_list, emotion_override)
    backend_warning = ""
    error_detail = ""
    async with gpu_lock:
        try:
            prompt_text = await run_text_completion(prompt_request, VOXTRAL_PROMPT_MAX_TOKENS)
        except TimeoutError as exc:
            prompt_text = fallback_dialog_prompt(language, prompt_type, history_list, dataset, emotion_override)
            backend = "fallback-timeout"
            error_detail = str(exc)
            backend_warning = f"Voxtral Timeout: {error_detail}"
            logger.warning("dialog/prompt fallback due to timeout: %s", error_detail)
        except Exception as exc:
            prompt_text = fallback_dialog_prompt(language, prompt_type, history_list, dataset, emotion_override)
            backend = f"fallback-error:{type(exc).__name__}"
            error_detail = str(exc)
            backend_warning = f"Voxtral Fehler ({type(exc).__name__}): {error_detail}"
            logger.warning("dialog/prompt fallback due to error [%s]: %s", type(exc).__name__, error_detail)
        else:
            backend = "voxtral"
            logger.info("dialog/prompt generated via voxtral for dataset=%s prompt_type=%s prompt_length=%s", dataset, prompt_type, prompt_length)

    return JSONResponse(
        content={
            "success": True,
            "promptText": prompt_text,
            "promptType": prompt_type,
            "promptLength": prompt_length,
            "backend": backend,
            "warning": backend_warning,
            "errorDetail": error_detail,
        }
    )


@app.post("/dialog/speak")
async def dialog_speak(request: Request) -> Response:
    payload = await request.json()
    text = (payload.get("text") or "").strip()
    voice = (payload.get("voice") or PIPER_DEFAULT_VOICE).strip()
    if not text:
        return JSONResponse(status_code=400, content={"success": False, "error": "Text required"})

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(f"{PIPER_URL}/synthesize", json={"text": text, "voice": voice})
            response.raise_for_status()
            headers = {"X-Sprachwerker-Voice": voice}
            return Response(content=response.content, media_type="audio/wav", headers=headers)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"success": False, "error": str(exc)})


@app.get("/segment-review/datasets")
async def segment_review_datasets(language: str = SOURCE_LANGUAGE_CODE) -> JSONResponse:
    return JSONResponse(content={"success": True, "datasets": list_datasets(language)})


@app.get("/segment-review/profiles")
async def segment_review_profiles(language: str = SOURCE_LANGUAGE_CODE) -> JSONResponse:
    return JSONResponse(content={"success": True, "profiles": list_dataset_profiles(language)})


@app.post("/segment-review/profiles")
async def create_dataset_profile(request: Request) -> JSONResponse:
    payload = await request.json()
    language = payload.get("language") or SOURCE_LANGUAGE_CODE
    dataset = sanitize_dataset_name(payload.get("dataset"))
    workflow = payload.get("workflow") or {}
    if not dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset name required"})

    profiles = load_dataset_profiles(language)
    if dataset in profiles:
        return JSONResponse(status_code=409, content={"success": False, "error": "Dataset already exists"})

    task_mode = workflow.get("taskMode") or DEFAULT_TASK_MODE
    speaker_profile = workflow.get("speakerProfile") or DEFAULT_SPEAKER_PROFILE
    if task_mode != "asr" and speaker_profile == "multi_speaker":
        speaker_profile = "single_speaker"

    profile = {
        "taskMode": task_mode,
        "audioDomain": workflow.get("audioDomain") or DEFAULT_AUDIO_DOMAIN,
        "targetEngine": workflow.get("targetEngine") or DEFAULT_TARGET_ENGINE,
        "speakerProfile": speaker_profile,
        "dialectLabel": (workflow.get("dialectLabel") or DEFAULT_DIALECT_LABEL).strip(),
        "emotionLabel": (workflow.get("emotionLabel") or DEFAULT_EMOTION_LABEL).strip(),
        "recordingQuality": workflow.get("recordingQuality") or DEFAULT_RECORDING_QUALITY,
        "ttsSuitability": workflow.get("ttsSuitability") or DEFAULT_TTS_SUITABILITY,
        "createdAt": int(Path(__file__).stat().st_mtime),
    }
    profiles[dataset] = profile
    save_dataset_profiles(language, profiles)
    return JSONResponse(content={"success": True, "dataset": dataset, "profile": profile})


@app.delete("/segment-review/profiles")
async def delete_dataset_profile(request: Request) -> JSONResponse:
    payload = await request.json()
    language = payload.get("language") or SOURCE_LANGUAGE_CODE
    dataset = sanitize_dataset_name(payload.get("dataset"))
    if not dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset name required"})
    profiles = load_dataset_profiles(language)
    if dataset not in profiles:
        return JSONResponse(status_code=404, content={"success": False, "error": "Dataset not found"})
    del profiles[dataset]
    save_dataset_profiles(language, profiles)
    return JSONResponse(content={"success": True, "dataset": dataset})


@app.get("/segment-review/checkup")
async def segment_review_checkup(language: str = SOURCE_LANGUAGE_CODE, ui_lang: str = UI_DEFAULT_LANG) -> JSONResponse:
    return JSONResponse(content={"success": True, "checkup": build_checkup(language, ui_lang), "uiLanguage": normalize_ui_lang(ui_lang)})


@app.post("/free-record/split")
async def free_record_split(
    audio: UploadFile = File(...),
    min_silence_ms: int = Form(700),
    silence_thresh_db: float = Form(-35.0),
) -> JSONResponse:
    """Split a free recording into segments by silence detection."""
    audio_bytes = await audio.read()
    try:
        wav_bytes, total_duration = normalize_audio(audio_bytes)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Audio error: {exc}"})

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        src_path = Path(tmp.name)

    try:
        # Run ffmpeg silencedetect
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-i", str(src_path),
                "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_ms / 1000:.3f}",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stderr = result.stderr

        # Parse silence intervals
        silence_starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", stderr)]
        silence_ends = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", stderr)]

        # Build cut points: midpoints of silence intervals
        cut_points = [0.0]
        for s, e in zip(silence_starts, silence_ends):
            cut_points.append((s + e) / 2.0)
        cut_points.append(total_duration)

        # Extract each segment as base64 WAV
        segments = []
        for i in range(len(cut_points) - 1):
            seg_start = cut_points[i]
            seg_end = cut_points[i + 1]
            seg_duration = seg_end - seg_start
            if seg_duration < 0.5:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_tmp:
                seg_path = Path(seg_tmp.name)

            try:
                seg_result = subprocess.run(
                    [
                        "ffmpeg", "-nostdin", "-y",
                        "-i", str(src_path),
                        "-ss", f"{seg_start:.3f}",
                        "-to", f"{seg_end:.3f}",
                        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                        str(seg_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if seg_result.returncode != 0 or not seg_path.exists():
                    continue
                seg_bytes = seg_path.read_bytes()
                seg_b64 = base64.b64encode(seg_bytes).decode()
                segments.append({
                    "index": len(segments),
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "duration": round(seg_duration, 3),
                    "audio": f"data:audio/wav;base64,{seg_b64}",
                })
            finally:
                seg_path.unlink(missing_ok=True)

        return JSONResponse(content={"success": True, "segments": segments, "totalDuration": round(total_duration, 3)})

    except Exception as exc:
        logger.warning("free_record_split failed: %s", exc)
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        src_path.unlink(missing_ok=True)


@app.post("/segment-review/migrate-legacy")
async def segment_review_migrate_legacy(request: Request) -> JSONResponse:
    payload = await request.json()
    language = payload.get("language") or SOURCE_LANGUAGE_CODE
    dataset = payload.get("dataset") or ""
    result = migrate_legacy_dataset(language, dataset)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(status_code=status_code, content=result)


@app.get("/segment-review/audio/{language}/{dataset}/{filename}", response_class=FileResponse)
async def segment_review_audio(language: str, dataset: str, filename: str) -> Response:
    safe_dataset = sanitize_dataset_name(dataset)
    safe_filename = Path(filename).name  # strip any path components
    if not safe_filename or safe_filename != filename:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid filename"})
    file_path = (dataset_dir(language, safe_dataset) / safe_filename).resolve()
    base = dataset_dir(language, safe_dataset).resolve()
    if not str(file_path).startswith(str(base)):
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid filename"})
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"success": False, "error": "File not found"})
    return FileResponse(file_path)


@app.post("/segment-review/save")
async def segment_review_save(request: Request) -> JSONResponse:
    payload = await request.json()
    language = payload.get("language") or SOURCE_LANGUAGE_CODE
    dataset = sanitize_dataset_name(payload.get("dataset"))
    items = payload.get("items") or []
    profiles = load_dataset_profiles(language)
    profile = profiles.get(dataset)

    if not dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset name required"})
    if not profile:
        return JSONResponse(status_code=400, content={"success": False, "error": "Dataset profile missing"})
    if not items:
        return JSONResponse(status_code=400, content={"success": False, "error": "No items provided"})

    saved_items = []
    saved_by_dataset: dict[str, int] = {}

    for item in items:
        text = (item.get("text") or "").strip()
        audio_base64 = item.get("audioBase64") or ""
        audio_ext = item.get("audioExt") or "wav"
        if audio_ext not in {"wav", "webm"}:
            audio_ext = "wav"
        if not text or not audio_base64:
            continue

        duration_seconds = 0.0
        try:
            if item.get("startTime") is not None and item.get("endTime") is not None:
                duration_seconds = max(0.0, float(item["endTime"]) - float(item["startTime"]))
        except (TypeError, ValueError):
            duration_seconds = 0.0
        bucket = classify_duration_bucket(duration_seconds)
        dataset_name = sanitize_dataset_name(f"{dataset}-{bucket}")
        group_dir = dataset_dir(language, dataset_name)
        group_dir.mkdir(parents=True, exist_ok=True)
        next_id = get_next_segment_id(group_dir)

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception:
            continue

        stem = f"{next_id:010d}"
        audio_path = group_dir / f"{stem}.{audio_ext}"
        text_path = group_dir / f"{stem}.txt"
        audio_path.write_bytes(audio_bytes)
        text_path.write_text(text, encoding="utf-8")

        metadata = {
            "id": stem,
            "text": text,
            "raw_text": item.get("rawText", ""),
            "start": item.get("startTime"),
            "end": item.get("endTime"),
            "duration_seconds": duration_seconds,
            "bucket": bucket,
            "base_dataset": dataset,
            "task_mode": profile.get("taskMode", DEFAULT_TASK_MODE),
            "audio_domain": profile.get("audioDomain", DEFAULT_AUDIO_DOMAIN),
            "target_engine": profile.get("targetEngine", DEFAULT_TARGET_ENGINE),
            "speaker_profile": profile.get("speakerProfile", DEFAULT_SPEAKER_PROFILE),
            "dialect_label": (item.get("dialectLabel") or profile.get("dialectLabel") or DEFAULT_DIALECT_LABEL).strip(),
            "style_label": (profile.get("styleLabel") or DEFAULT_STYLE_LABEL).strip(),
            "emotion_label": (item.get("emotionLabel") or profile.get("emotionLabel") or DEFAULT_EMOTION_LABEL).strip(),
            "recording_quality": profile.get("recordingQuality", DEFAULT_RECORDING_QUALITY),
            "tts_suitability": profile.get("ttsSuitability", DEFAULT_TTS_SUITABILITY),
        }
        manifest_path = group_dir / "_review_manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        saved_items.append({**metadata, "dataset": dataset_name})
        saved_by_dataset[dataset_name] = saved_by_dataset.get(dataset_name, 0) + 1

    return JSONResponse(
        content={
            "success": True,
            "saved": len(saved_items),
            "dataset": dataset,
            "datasets": saved_by_dataset,
            "items": saved_items,
        }
    )


@app.delete("/segment-review/datasets/{dataset}")
async def delete_dataset(dataset: str, language: str = SOURCE_LANGUAGE_CODE) -> JSONResponse:
    safe_dataset = sanitize_dataset_name(dataset)
    if not safe_dataset:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid dataset"})
    target_dir = dataset_dir(language, safe_dataset)
    if not target_dir.exists():
        return JSONResponse(status_code=404, content={"success": False, "error": "Dataset not found"})
    shutil.rmtree(target_dir)
    return JSONResponse(content={"success": True, "dataset": safe_dataset})


@app.delete("/segment-review/items/{dataset}/{item_id}")
async def delete_dataset_item(dataset: str, item_id: str, language: str = SOURCE_LANGUAGE_CODE) -> JSONResponse:
    safe_dataset = sanitize_dataset_name(dataset)
    safe_item_id = sanitize_item_id(item_id)
    if not safe_dataset or not safe_item_id:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid item"})

    group_dir = dataset_dir(language, safe_dataset)
    removed = False
    for path in (group_dir / f"{safe_item_id}.wav", group_dir / f"{safe_item_id}.webm", group_dir / f"{safe_item_id}.txt"):
        if path.exists():
            path.unlink()
            removed = True

    if not removed:
        return JSONResponse(status_code=404, content={"success": False, "error": "Item not found"})
    return JSONResponse(content={"success": True, "dataset": safe_dataset, "id": safe_item_id})


@app.post("/segment-review/items/{dataset}/{item_id}")
async def update_dataset_item(dataset: str, item_id: str, request: Request, language: str = SOURCE_LANGUAGE_CODE) -> JSONResponse:
    safe_dataset = sanitize_dataset_name(dataset)
    safe_item_id = sanitize_item_id(item_id)
    if not safe_dataset or not safe_item_id:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid item"})

    payload = await request.json()
    new_text = (payload.get("text") or "").strip()
    keep_marked = payload.get("keepMarked")
    if not new_text and keep_marked is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "Text required"})

    group_dir = dataset_dir(language, safe_dataset)
    text_path = group_dir / f"{safe_item_id}.txt"
    if not text_path.exists():
        return JSONResponse(status_code=404, content={"success": False, "error": "Item not found"})

    if new_text:
        text_path.write_text(new_text, encoding="utf-8")
    else:
        new_text = text_path.read_text(encoding="utf-8")

    manifest_path = group_dir / "_review_manifest.jsonl"
    if manifest_path.exists():
        updated_lines = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(entry.get("id")) == safe_item_id:
                entry["text"] = new_text
                if keep_marked is not None:
                    entry["keep_marked"] = bool(keep_marked)
            updated_lines.append(json.dumps(entry, ensure_ascii=False))
        manifest_path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")

    return JSONResponse(content={"success": True, "dataset": safe_dataset, "id": safe_item_id, "text": new_text, "keepMarked": bool(keep_marked) if keep_marked is not None else None})


@app.get("/health")
async def health() -> dict[str, Any]:
    piper_ok = False
    piper_error = ""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{PIPER_URL}/health")
            response.raise_for_status()
            piper_ok = True
    except Exception as exc:
        piper_error = str(exc)
    return {
        "status": "ok",
        "app_title": APP_TITLE,
        "dialect_name": DIALECT_NAME,
        "output_dir": str(OUTPUT_DIR),
        "cli_path": CLI_PATH,
        "text_cli_path": VOXTRAL_TEXT_CLI_PATH,
        "model_path": MODEL_PATH,
        "mmproj_path": MMPROJ_PATH,
        "cli_exists": Path(CLI_PATH).exists(),
        "text_cli_exists": Path(VOXTRAL_TEXT_CLI_PATH).exists(),
        "model_exists": Path(MODEL_PATH).exists(),
        "mmproj_exists": Path(MMPROJ_PATH).exists(),
        "piper_url": PIPER_URL,
        "piper_ok": piper_ok,
        "piper_error": piper_error,
    }


if __name__ == "__main__":
    ensure_dirs()
    port = int(os.environ.get("PORT", "8095"))
    uvicorn.run(app, host="0.0.0.0", port=port)
