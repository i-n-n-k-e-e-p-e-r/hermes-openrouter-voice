"""Catalog invariants — the lists that decide whether a request is even valid.

Stdlib-only except for the provider-shaped test, which skips when Hermes is not importable (the
plugin's own tests must run outside a Hermes checkout too).

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_PLUGIN = _HERE.parent


def _plugin_module(name: str):
    """Load a plugin module by path (the directory name has a hyphen, so it is not a package)."""
    module_name = f"hermes_openrouter_voice_{name}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, _PLUGIN / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _plugin_module("common")


def test_catalogs_are_populated_and_unique(common):
    for label, slugs in (("STT_CATALOG", common.STT_CATALOG), ("TTS_CATALOG", common.TTS_CATALOG),
                         ("TTS_VOICES", common.TTS_VOICES)):
        assert slugs, f"{label} is empty"
        assert len(slugs) == len(set(slugs)), f"{label} has duplicates"


def test_every_model_slug_is_vendor_prefixed(common):
    """A bare name (`whisper-1`) is a guaranteed 400 — the API only takes `vendor/model`."""
    for slug in (*common.STT_CATALOG, *common.TTS_CATALOG):
        assert "/" in slug, f"{slug!r} is not a vendor-prefixed slug"


def test_defaults_are_in_their_catalogs(common):
    assert common.DEFAULT_STT_MODEL in common.STT_CATALOG
    assert common.DEFAULT_TTS_MODEL in common.TTS_CATALOG
    assert common.DEFAULT_TTS_VOICE in common.TTS_VOICES


def test_every_tts_model_has_a_voice_entry(common):
    """The map is the point of per-model voices: a model with no entry would silently offer the
    default model's voices (a 400 waiting to happen). Empty tuples are valid — the models API
    publishes no list for the fish-audio family, which speak with the provider default."""
    missing = [model for model in common.TTS_CATALOG if model not in common.TTS_VOICES_BY_MODEL]
    assert not missing, f"no voice entry for {missing}"


def test_voice_sets_are_unique_per_model(common):
    for model, voices in common.TTS_VOICES_BY_MODEL.items():
        assert len(voices) == len(set(voices)), f"{model} lists a voice twice"


def test_default_voice_belongs_to_the_default_model(common):
    assert common.DEFAULT_TTS_VOICE in common.voices_for_model(common.DEFAULT_TTS_MODEL)
    assert common.DEFAULT_TTS_VOICE in common.TTS_VOICES


def test_voices_for_model_is_total(common):
    """Unknown models must return empty rather than raising or leaking another model's voices."""
    assert common.voices_for_model("nope/not-a-model") == ()
    assert common.voices_for_model("") == ()


def test_resolve_tts_settings_reads_openrouter_file_format(common, monkeypatch):
    monkeypatch.setattr(common, "_section", lambda kind, explicit="": {
        "openrouter": {"file_format": "pcm"}
    } if kind == "tts" else {})
    monkeypatch.setattr(common, "api_key", lambda: "test-key")

    settings = common.resolve_tts_settings()

    assert settings["file_format"] == "pcm"


def test_providers_expose_the_catalogs_to_the_catalog_surface():
    """`hermes tools` / the dashboard list models via the provider, not the module constants."""
    pytest.importorskip("agent.transcription_provider", reason="needs a Hermes environment")
    # Dependency order matters (the same trap register() has): the wrappers import the cores by name.
    _plugin_module("common")
    _plugin_module("stt_core")
    _plugin_module("tts_core")
    stt = _plugin_module("stt")
    tts = _plugin_module("tts")

    stt_models = [entry["id"] for entry in stt.build_provider().list_models()]
    tts_models = [entry["id"] for entry in tts.build_provider().list_models()]

    assert stt_models and tts_models
    assert stt.build_provider().default_model() in stt_models
    assert tts.build_provider().default_model() in tts_models
    assert tts.build_provider().default_voice() in [v["id"] for v in tts.build_provider().list_voices()]
