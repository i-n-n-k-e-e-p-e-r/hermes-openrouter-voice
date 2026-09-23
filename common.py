"""Shared helpers for the OpenRouter voice plugin (speech-to-text + text-to-speech).

Nothing here imports Hermes at module import time beyond a best-effort config read, so the same
module serves the plugin entry point and the two command-provider shims.

Endpoint facts established by probing the live API (2026-09-14), baked in below:

* Transcription accepts only ``response_format: json|verbose_json`` — ``"text"`` is a 400.
* 19 of 20 transcription models accept MP3/WebM; ``meta/muse-voice-transcribe-1.0`` requires
  16 kHz or 24 kHz mono RIFF/WAVE (a 22.05 kHz ``say`` recording is a live 400) — so PCM WAV input is
  normalised in pure Python and other containers go through ffmpeg when it is available.
* Speech models are vendor-prefixed catalog slugs, and voices are **model-specific**.
"""

from __future__ import annotations

import array
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any, Dict, List, Optional

ENV_FILE = pathlib.Path.home() / ".hermes" / ".env"
WAV_TARGET_RATE = 16000  # Meta's transcription endpoint accepts 16000 or 24000 Hz only

DEFAULT_STT_MODEL = os.environ.get("OPENROUTER_STT_MODEL", "openai/whisper-large-v3")
DEFAULT_STT_BASE_URL = os.environ.get("OPENROUTER_STT_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_TTS_MODEL = os.environ.get("TTS_OPENROUTER_MODEL", "deepgram/aura-2")
DEFAULT_TTS_VOICE = os.environ.get("TTS_OPENROUTER_VOICE", "aura-2-thalia-en")
DEFAULT_TTS_BASE_URL = os.environ.get("TTS_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Speed, measured live 2026-09-17 by comparing audio length at speed=1.5 against the default:
#   * these honour it natively (ratio 0.60-0.66)
#   * voxtral/orpheus/grok silently ignore it (ratio 0.89-0.98)
#   * qwen rejects the parameter outright: HTTP 400 "does not support the speed parameter"
# So `speed` is only sent to models measured to honour it; every other model gets the parameter
# omitted and the audio time-stretched locally instead, which cannot 400 and cannot silently no-op.
# Native ranges, probed 2026-09-17 by sending 0.25/0.5/0.7/1.0/1.5/1.6/2.0/3.0:
#   deepgram/aura-2            accepts 0.7-1.5 only; 1.6 and 2.0 are HTTP 400
#   minimax, mai-voice-2, kokoro accept 0.25-3.0
# Anything outside a model's range is sent as the nearest in-range value with the remainder
# time-stretched locally, so one control covers 0.25-4.0 without ever 400ing.
NATIVE_SPEED_RANGES: Dict[str, tuple] = {
    "deepgram/aura-2": (0.7, 1.5),
    "minimax/speech-2.8-turbo": (0.25, 3.0),
    "microsoft/mai-voice-2": (0.25, 3.0),
    "hexgrad/kokoro-82m": (0.25, 3.0),
}

# Every transcription model on OpenRouter when this was written; feeds ``list_models()`` and the CLI
# pickers, which take slugs verbatim.
STT_CATALOG = (
    "openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "openai/whisper-1",
    "openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe", "openai/gpt-transcribe",
    "mistralai/voxtral-mini-transcribe", "google/chirp-3", "deepgram/nova-3",
    "x-ai/grok-stt-1.0", "qwen/qwen3-asr-flash-2026-02-10", "microsoft/mai-transcribe-2",
    "nvidia/parakeet-tdt-0.6b-v3", "fish-audio/transcribe-1", "meta/muse-voice-transcribe-1.0",
    "mistralai/voxtral-small-24b-2507-stt", "mistralai/voxtral-mini-3b-2507",
    "qwen/qwen3-asr-1.7b", "qwen/qwen3-asr-0.6b",
    "microsoft/mai-transcribe-1.5", "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
)

# Every speech (TTS) model OpenRouter listed when this was written; feeds ``list_models()`` so the
# model is discoverable instead of guessed. Refresh with ``python models.py --live``.
TTS_CATALOG = (
    "deepgram/aura-2", "deepgram/flux-tts:free", "minimax/speech-2.8-turbo", "minimax/speech-2.8-hd",
    "fish-audio/s2.1-pro", "fish-audio/s2.1-pro-free:free", "fish-audio/s2-pro", "fish-audio/s1",
    "microsoft/mai-voice-2", "microsoft/mai-voice-2-flash", "qwen/qwen-audio-3.0-tts-flash",
    "qwen/qwen-audio-3.0-tts-plus", "x-ai/grok-voice-tts-1.0", "google/gemini-3.1-flash-tts-preview",
    "canopylabs/orpheus-3b-0.1-ft", "sesame/csm-1b", "hexgrad/kokoro-82m",
    "mistralai/voxtral-mini-tts-2603",
)

# Voices are MODEL-SPECIFIC: a mismatched pair is a 400 that names the model, so the plugin
# ships each model's published set (the models API exposes `supported_voices`; there is no
# voices endpoint). Refresh with `python models.py --live --voices`.
TTS_VOICES_BY_MODEL = {
    "canopylabs/orpheus-3b-0.1-ft": (
        "tara",
        "leah",
        "jess",
        "leo",
        "dan",
        "mia",
        "zac",
    ),
    "deepgram/aura-2": (
        "aura-2-thalia-en",
        "aura-2-agathe-fr",
        "aura-2-agustina-es",
        "aura-2-alvaro-es",
        "aura-2-ama-ja",
        "aura-2-amalthea-en",
        "aura-2-andromeda-en",
        "aura-2-antonia-es",
        "aura-2-apollo-en",
        "aura-2-aquila-es",
        "aura-2-arcas-en",
        "aura-2-aries-en",
        "aura-2-asteria-en",
        "aura-2-athena-en",
        "aura-2-atlas-en",
        "aura-2-aurelia-de",
        "aura-2-aurora-en",
        "aura-2-beatrix-nl",
        "aura-2-callista-en",
        "aura-2-carina-es",
        "aura-2-celeste-es",
        "aura-2-cesare-it",
        "aura-2-cinzia-it",
        "aura-2-cora-en",
        "aura-2-cordelia-en",
        "aura-2-cornelia-nl",
        "aura-2-daphne-nl",
        "aura-2-delia-en",
        "aura-2-demetra-it",
        "aura-2-diana-es",
        "aura-2-dionisio-it",
        "aura-2-draco-en",
        "aura-2-ebisu-ja",
        "aura-2-elara-de",
        "aura-2-electra-en",
        "aura-2-elio-it",
        "aura-2-estrella-es",
        "aura-2-fabian-de",
        "aura-2-flavio-it",
        "aura-2-fujin-ja",
        "aura-2-gloria-es",
        "aura-2-harmonia-en",
        "aura-2-hector-fr",
        "aura-2-helena-en",
        "aura-2-hera-en",
        "aura-2-hermes-en",
        "aura-2-hestia-nl",
        "aura-2-hyperion-en",
        "aura-2-iris-en",
        "aura-2-izanami-ja",
        "aura-2-janus-en",
        "aura-2-javier-es",
        "aura-2-julius-de",
        "aura-2-juno-en",
        "aura-2-jupiter-en",
        "aura-2-kara-de",
        "aura-2-lara-de",
        "aura-2-lars-nl",
        "aura-2-leda-nl",
        "aura-2-livia-it",
        "aura-2-luciano-es",
        "aura-2-luna-en",
        "aura-2-maia-it",
        "aura-2-mars-en",
        "aura-2-melia-it",
        "aura-2-minerva-en",
        "aura-2-neptune-en",
        "aura-2-nestor-es",
        "aura-2-odysseus-en",
        "aura-2-olivia-es",
        "aura-2-ophelia-en",
        "aura-2-orion-en",
        "aura-2-orpheus-en",
        "aura-2-pandora-en",
        "aura-2-phoebe-en",
        "aura-2-pluto-en",
        "aura-2-rhea-nl",
        "aura-2-roman-nl",
        "aura-2-sander-nl",
        "aura-2-saturn-en",
        "aura-2-selena-es",
        "aura-2-selene-en",
        "aura-2-silvia-es",
        "aura-2-sirio-es",
        "aura-2-theia-en",
        "aura-2-uzume-ja",
        "aura-2-valerio-es",
        "aura-2-vesta-en",
        "aura-2-viktoria-de",
        "aura-2-zeus-en",
    ),
    "deepgram/flux-tts:free": (
        "flux-alexis-en",
        "flux-bree-en",
        "flux-brittany-en",
        "flux-brooke-en",
        "flux-bruce-en",
        "flux-cliff-en",
        "flux-cole-en",
        "flux-colin-en",
        "flux-conor-en",
        "flux-donovan-en",
        "flux-drew-en",
        "flux-elise-en",
        "flux-gemma-en",
        "flux-haley-en",
        "flux-hannah-en",
        "flux-heather-en",
        "flux-jack-en",
        "flux-kai-en",
        "flux-kelsey-en",
        "flux-kit-en",
        "flux-maeve-en",
        "flux-marcelo-en",
        "flux-marcus-en",
        "flux-meena-en",
        "flux-meghan-en",
        "flux-miles-en",
        "flux-naveen-en",
        "flux-paige-en",
        "flux-priya-en",
        "flux-rufus-en",
        "flux-sean-en",
        "flux-sharon-en",
        "flux-sienna-en",
        "flux-tanner-en",
        "flux-wade-en",
        "flux-wes-en",
    ),
    "fish-audio/s1": (),  # no list published; the provider default voice works
    "fish-audio/s2-pro": (),  # no list published; the provider default voice works
    "fish-audio/s2.1-pro": (),  # no list published; the provider default voice works
    "fish-audio/s2.1-pro-free:free": (),  # no list published; the provider default voice works
    "google/gemini-3.1-flash-tts-preview": (
        "Zephyr",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    ),
    "hexgrad/kokoro-82m": (
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ),
    "microsoft/mai-voice-2": (
        "en-US-Harper:MAI-Voice-2",
        "es-MX-Valeria:MAI-Voice-2",
        "fr-FR-Soleil:MAI-Voice-2",
        "de-DE-Klaus:MAI-Voice-2",
    ),
    "microsoft/mai-voice-2-flash": (
        "en-US-Harper:MAI-Voice-2",
        "es-MX-Valeria:MAI-Voice-2",
        "fr-FR-Soleil:MAI-Voice-2",
        "de-DE-Klaus:MAI-Voice-2",
    ),
    "minimax/speech-2.8-hd": (
        "English_expressive_narrator",
        "English_radiant_girl",
        "English_magnetic_voiced_man",
        "English_compelling_lady1",
        "English_Aussie_Bloke",
        "English_captivating_female1",
        "English_Upbeat_Woman",
        "English_Trustworth_Man",
        "English_CalmWoman",
        "English_UpsetGirl",
        "English_Gentle-voiced_man",
        "English_Whispering_girl",
        "English_Diligent_Man",
        "English_Graceful_Lady",
        "English_ReservedYoungMan",
        "English_PlayfulGirl",
        "English_ManWithDeepVoice",
        "English_MaturePartner",
        "English_FriendlyPerson",
        "English_MatureBoss",
        "English_Debator",
        "English_LovelyGirl",
        "English_Steadymentor",
        "English_Deep-VoicedGentleman",
        "English_Wiselady",
        "English_CaptivatingStoryteller",
        "English_DecentYoungMan",
        "English_SentimentalLady",
        "English_ImposingManner",
        "English_SadTeen",
        "English_PassionateWarrior",
        "English_WiseScholar",
        "English_Soft-spokenGirl",
        "English_SereneWoman",
        "English_ConfidentWoman",
        "English_PatientMan",
        "English_Comedian",
        "English_BossyLeader",
        "English_Strong-WilledBoy",
        "English_StressedLady",
        "English_AssertiveQueen",
        "English_AnimeCharacter",
        "English_Jovialman",
        "English_WhimsicalGirl",
        "English_Kind-heartedGirl",
    ),
    "minimax/speech-2.8-turbo": (
        "English_expressive_narrator",
        "English_radiant_girl",
        "English_magnetic_voiced_man",
        "English_compelling_lady1",
        "English_Aussie_Bloke",
        "English_captivating_female1",
        "English_Upbeat_Woman",
        "English_Trustworth_Man",
        "English_CalmWoman",
        "English_UpsetGirl",
        "English_Gentle-voiced_man",
        "English_Whispering_girl",
        "English_Diligent_Man",
        "English_Graceful_Lady",
        "English_ReservedYoungMan",
        "English_PlayfulGirl",
        "English_ManWithDeepVoice",
        "English_MaturePartner",
        "English_FriendlyPerson",
        "English_MatureBoss",
        "English_Debator",
        "English_LovelyGirl",
        "English_Steadymentor",
        "English_Deep-VoicedGentleman",
        "English_Wiselady",
        "English_CaptivatingStoryteller",
        "English_DecentYoungMan",
        "English_SentimentalLady",
        "English_ImposingManner",
        "English_SadTeen",
        "English_PassionateWarrior",
        "English_WiseScholar",
        "English_Soft-spokenGirl",
        "English_SereneWoman",
        "English_ConfidentWoman",
        "English_PatientMan",
        "English_Comedian",
        "English_BossyLeader",
        "English_Strong-WilledBoy",
        "English_StressedLady",
        "English_AssertiveQueen",
        "English_AnimeCharacter",
        "English_Jovialman",
        "English_WhimsicalGirl",
        "English_Kind-heartedGirl",
    ),
    "mistralai/voxtral-mini-tts-2603": (
        "en_paul_sad",
        "en_paul_neutral",
        "en_paul_happy",
        "en_paul_frustrated",
        "en_paul_excited",
        "en_paul_confident",
        "en_paul_cheerful",
        "en_paul_angry",
        "gb_oliver_neutral",
        "gb_oliver_sad",
        "gb_oliver_excited",
        "gb_oliver_curious",
        "gb_oliver_confident",
        "gb_oliver_cheerful",
        "gb_oliver_angry",
        "gb_jane_sarcasm",
        "gb_jane_confused",
        "gb_jane_shameful",
        "gb_jane_sad",
        "gb_jane_neutral",
        "gb_jane_jealousy",
        "gb_jane_frustrated",
        "gb_jane_curious",
        "gb_jane_confident",
        "fr_marie_sad",
        "fr_marie_neutral",
        "fr_marie_happy",
        "fr_marie_excited",
        "fr_marie_curious",
        "fr_marie_angry",
    ),
    "qwen/qwen-audio-3.0-tts-flash": (
        "loongjohn",
        "longanhuan_v3.6",
    ),
    "qwen/qwen-audio-3.0-tts-plus": (
        "longanlingxin",
        "longanlufeng",
    ),
    "sesame/csm-1b": (
        "conversational_a",
        "conversational_b",
        "read_speech_a",
        "read_speech_b",
        "read_speech_c",
        "read_speech_d",
        "none",
    ),
    "x-ai/grok-voice-tts-1.0": (
        "eve",
        "ara",
        "rex",
        "sal",
        "leo",
    ),
}

# The default model's full set, default voice first — what `list_voices()` reports.
TTS_VOICES = (
    "aura-2-thalia-en",
    "aura-2-agathe-fr",
    "aura-2-agustina-es",
    "aura-2-alvaro-es",
    "aura-2-ama-ja",
    "aura-2-amalthea-en",
    "aura-2-andromeda-en",
    "aura-2-antonia-es",
    "aura-2-apollo-en",
    "aura-2-aquila-es",
    "aura-2-arcas-en",
    "aura-2-aries-en",
    "aura-2-asteria-en",
    "aura-2-athena-en",
    "aura-2-atlas-en",
    "aura-2-aurelia-de",
    "aura-2-aurora-en",
    "aura-2-beatrix-nl",
    "aura-2-callista-en",
    "aura-2-carina-es",
    "aura-2-celeste-es",
    "aura-2-cesare-it",
    "aura-2-cinzia-it",
    "aura-2-cora-en",
    "aura-2-cordelia-en",
    "aura-2-cornelia-nl",
    "aura-2-daphne-nl",
    "aura-2-delia-en",
    "aura-2-demetra-it",
    "aura-2-diana-es",
    "aura-2-dionisio-it",
    "aura-2-draco-en",
    "aura-2-ebisu-ja",
    "aura-2-elara-de",
    "aura-2-electra-en",
    "aura-2-elio-it",
    "aura-2-estrella-es",
    "aura-2-fabian-de",
    "aura-2-flavio-it",
    "aura-2-fujin-ja",
    "aura-2-gloria-es",
    "aura-2-harmonia-en",
    "aura-2-hector-fr",
    "aura-2-helena-en",
    "aura-2-hera-en",
    "aura-2-hermes-en",
    "aura-2-hestia-nl",
    "aura-2-hyperion-en",
    "aura-2-iris-en",
    "aura-2-izanami-ja",
    "aura-2-janus-en",
    "aura-2-javier-es",
    "aura-2-julius-de",
    "aura-2-juno-en",
    "aura-2-jupiter-en",
    "aura-2-kara-de",
    "aura-2-lara-de",
    "aura-2-lars-nl",
    "aura-2-leda-nl",
    "aura-2-livia-it",
    "aura-2-luciano-es",
    "aura-2-luna-en",
    "aura-2-maia-it",
    "aura-2-mars-en",
    "aura-2-melia-it",
    "aura-2-minerva-en",
    "aura-2-neptune-en",
    "aura-2-nestor-es",
    "aura-2-odysseus-en",
    "aura-2-olivia-es",
    "aura-2-ophelia-en",
    "aura-2-orion-en",
    "aura-2-orpheus-en",
    "aura-2-pandora-en",
    "aura-2-phoebe-en",
    "aura-2-pluto-en",
    "aura-2-rhea-nl",
    "aura-2-roman-nl",
    "aura-2-sander-nl",
    "aura-2-saturn-en",
    "aura-2-selena-es",
    "aura-2-selene-en",
    "aura-2-silvia-es",
    "aura-2-sirio-es",
    "aura-2-theia-en",
    "aura-2-uzume-ja",
    "aura-2-valerio-es",
    "aura-2-vesta-en",
    "aura-2-viktoria-de",
    "aura-2-zeus-en",
)


def voices_for_model(model: str) -> tuple:
    """Voice IDs the given model accepts; empty when the catalog publishes none.

    Some models (the fish-audio family) list no voices and instead speak with the provider's
    default voice, which is why this can legitimately be empty.
    """
    return TTS_VOICES_BY_MODEL.get((model or "").strip(), ())

FFMPEG_CANDIDATES = (
    "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "/snap/bin/ffmpeg",
)


# --------------------------------------------------------------------------- credentials + config

def api_key() -> str:
    """``OPENROUTER_API_KEY`` from the environment, else ``~/.hermes/.env`` (the chat key)."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def config_candidates(explicit: str = "") -> List[pathlib.Path]:
    """Config files to try, most specific first.

    Profiles do not always differ by ``HERMES_HOME`` (the desktop passes ``--profile <name>``), so an
    implicit lookup can read the wrong profile's config — the shims therefore accept ``--config``.
    """
    if explicit:
        return [pathlib.Path(explicit).expanduser()]
    candidates: List[pathlib.Path] = []
    env_config = os.environ.get("HERMES_CONFIG", "").strip()
    if env_config:
        candidates.append(pathlib.Path(env_config).expanduser())
    profile = os.environ.get("HERMES_PROFILE", "").strip()
    if profile:
        candidates.append(pathlib.Path.home() / ".hermes" / "profiles" / profile / "config.yaml")
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        candidates.append(pathlib.Path(hermes_home).expanduser() / "config.yaml")
    candidates.append(pathlib.Path.home() / ".hermes" / "config.yaml")
    return candidates


def load_config(explicit: str = "") -> Dict[str, Any]:
    """The active (or explicitly named) config, via Hermes' loader when importable."""
    if not explicit:
        try:
            from hermes_cli.config import load_config as _load

            data = _load()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        import yaml
    except ImportError:
        return {}
    for path in config_candidates(explicit):
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _section(kind: str, explicit: str = "") -> Dict[str, Any]:
    section = load_config(explicit).get(kind)
    return section if isinstance(section, dict) else {}


def _per_provider(section: Dict[str, Any], key: str = "openrouter") -> Dict[str, Any]:
    block = section.get(key)
    return block if isinstance(block, dict) else {}


def _command_block(section: Dict[str, Any], key: str = "openrouter") -> Dict[str, Any]:
    providers = section.get("providers")
    if isinstance(providers, dict):
        block = providers.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_stt_settings(explicit_config: str = "") -> Dict[str, str]:
    section = _section("stt", explicit_config)
    per_provider = _per_provider(section)
    return {
        "model": _first_string(per_provider.get("model"), _command_block(section).get("model"))
                 or DEFAULT_STT_MODEL,
        "language": _first_string(per_provider.get("language"), section.get("language")),
        "base_url": _first_string(per_provider.get("base_url")).rstrip("/") or DEFAULT_STT_BASE_URL,
        "api_key": api_key(),
    }


def resolve_tts_settings(explicit_config: str = "") -> Dict[str, Any]:
    section = _section("tts", explicit_config)
    per_provider = _per_provider(section)
    speed: Any = per_provider.get("speed")
    if not isinstance(speed, (int, float)):
        speed = section.get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    return {
        "model": _first_string(per_provider.get("model"), _command_block(section).get("model"))
                 or DEFAULT_TTS_MODEL,
        "voice": _first_string(per_provider.get("voice"), _command_block(section).get("voice"))
                 or DEFAULT_TTS_VOICE,
        "file_format": _first_string(per_provider.get("file_format")).lower() or "mp3",
        "base_url": _first_string(per_provider.get("base_url")).rstrip("/") or DEFAULT_TTS_BASE_URL,
        "speed": speed,
        "api_key": api_key(),
    }


# --------------------------------------------------------------------------- audio preparation

def ffmpeg_path() -> Optional[str]:
    """PATH first, then the usual install prefixes — the app-spawned backend can have a short PATH.

    ffmpeg is not installed by default on macOS, and the desktop records ``audio/webm;codecs=opus``,
    so a WAV-only model needs it (``brew install ffmpeg``) for that container.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _pcm16_mono_downmix(frames: bytes, channels: int) -> bytes:
    if channels <= 1:
        return frames
    samples = array.array("h")
    samples.frombytes(frames if len(frames) % 2 == 0 else frames[:-1])
    if sys.byteorder == "big":
        samples.byteswap()
    mono = array.array("h", bytes(2 * (len(samples) // channels)))
    for frame in range(len(mono)):
        base = frame * channels
        total = 0
        for channel in range(channels):
            total += samples[base + channel]
        mono[frame] = int(total / channels)
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes()


def _pcm16_resample(frames: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear interpolation — speech-grade, and free of any audioop/ffmpeg dependency."""
    if in_rate == out_rate or not frames:
        return frames
    samples = array.array("h")
    samples.frombytes(frames if len(frames) % 2 == 0 else frames[:-1])
    if sys.byteorder == "big":
        samples.byteswap()
    ratio = in_rate / out_rate
    out_count = max(1, int(len(samples) / ratio))
    out = array.array("h", bytes(2 * out_count))
    for index in range(out_count):
        position = index * ratio
        low = int(position)
        frac = position - low
        first = samples[low]
        second = samples[low + 1] if low + 1 < len(samples) else first
        out[index] = int(first + (second - first) * frac)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def normalize_wav(src: str) -> str:
    """Rewrite a PCM WAV as 16 kHz mono, in pure Python (no ffmpeg needed)."""
    try:
        with wave.open(src, "rb") as handle:
            channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError):
        return src
    if width != 2:  # only 16-bit PCM here; ffmpeg covers the rest
        return src
    if channels == 1 and rate == WAV_TARGET_RATE:
        return src
    frames = _pcm16_resample(_pcm16_mono_downmix(frames, channels), rate, WAV_TARGET_RATE)
    out = pathlib.Path(tempfile.mkdtemp()) / "audio.wav"
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(WAV_TARGET_RATE)
        handle.writeframes(frames)
    return str(out)


def prepare_audio(src: str) -> str:
    """Best-effort 16 kHz mono PCM; the input untouched when nothing can convert it."""
    if src.lower().endswith(".wav"):
        return normalize_wav(src)
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return src
    out = pathlib.Path(tempfile.mkdtemp()) / "audio.wav"
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(WAV_TARGET_RATE),
         "-f", "wav", str(out)],
        capture_output=True,
    )
    if proc.returncode == 0 and out.exists() and out.stat().st_size > 44:
        return str(out)
    return src


# --------------------------------------------------------------------------- HTTP

def _multipart(fields: Dict[str, str], file_field: str, file_path: str) -> tuple:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{os.path.basename(file_path)}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode()
    body += pathlib.Path(file_path).read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), boundary


def _headers(key: str, content_type: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
        "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
        "X-Title": "Hermes Agent",
    }


def _request(url: str, data: bytes, headers: Dict[str, str], timeout: int = 180) -> bytes:
    """POST and return the raw body; raise ``RuntimeError`` naming the endpoint error."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


def split_speed(model: str, rate: float) -> tuple:
    """Split *rate* into ``(native_speed, local_stretch)``; ``native_speed`` is None when unused.

    The product is the requested rate, so the caller only has to pass the native value to the API
    and stretch the result by the local factor.
    """
    rate = max(0.25, min(4.0, rate))
    if abs(rate - 1.0) < 0.01:
        return None, 1.0
    bounds = NATIVE_SPEED_RANGES.get(model)
    if not bounds:
        return None, rate
    low, high = bounds
    native = min(max(rate, low), high)
    local = rate / native
    return (native if abs(native - 1.0) > 0.01 else None,
            local if abs(local - 1.0) > 0.01 else 1.0)


def build_atempo_chain(rate: float) -> str:
    """ffmpeg ``atempo`` filter chain for *rate* (one instance covers 0.5-2.0, so chain the rest)."""
    parts: List[str] = []
    remaining = max(0.25, min(4.0, rate))
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.01:
        parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def time_stretch(path: str, rate: float) -> str:
    """Rewrite the audio at *rate* (pitch-preserving); the original path when ffmpeg is unavailable."""
    if abs(rate - 1.0) < 0.01:
        return path
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return path
    chain = build_atempo_chain(rate)
    if not chain:
        return path
    out = pathlib.Path(tempfile.mkdtemp()) / (pathlib.Path(path).stem + "-speed.mp3")
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", path, "-filter:a", chain, str(out)],
        capture_output=True,
    )
    if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return str(out)
    return path


def transcribe_request(audio_path: str, *, model: str, language: str, base_url: str, key: str) -> str:
    """POST audio to ``/audio/transcriptions`` and return the transcript text."""
    payload, boundary = _multipart(
        {"model": model, "response_format": "json", **({"language": language} if language else {})},
        "file",
        prepare_audio(audio_path),
    )
    raw = _request(
        f"{base_url}/audio/transcriptions", payload,
        _headers(key, f"multipart/form-data; boundary={boundary}"),
    )
    body = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.strip()
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"].strip()
    return body.strip()


def synthesize_request(text: str, *, model: str, voice: str, speed: Any, base_url: str, key: str,
                       instructions: str = "", response_format: str = "mp3") -> bytes:
    """POST to ``/audio/speech`` and return the audio bytes in the requested response format."""
    body: Dict[str, Any] = {"model": model, "input": text, "response_format": response_format}
    if voice and voice.strip():
        # Omitted, not empty: models that publish no voices speak with the provider default, and an
        # empty string fails their schema validation.
        body["voice"] = voice.strip()
    if isinstance(speed, (int, float)) and float(speed) != 1.0:
        body["speed"] = float(speed)
    if instructions:
        body["instructions"] = instructions
    data = json.dumps(body).encode("utf-8")
    return _request(f"{base_url}/audio/speech", data, _headers(key, "application/json"))
