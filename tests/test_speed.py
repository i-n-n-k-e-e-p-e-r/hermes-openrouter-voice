"""Speed routing and the atempo chain — the logic that stops `speed` from breaking some models.

Measured basis (2026-09-17): aura-2/minimax/mai-voice-2/kokoro/fish-audio honour `speed` natively;
voxtral/orpheus/grok ignore it; qwen **rejects** it with HTTP 400. So the parameter is sent only to
models known to honour it, and everything else is time-stretched locally.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import wave

import pytest

_PLUGIN = pathlib.Path(__file__).resolve().parent.parent


def _plugin_module(name: str):
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


@pytest.fixture()
def tts_core():
    _plugin_module("common")
    module = _plugin_module("tts_core")
    # default: pretend the API returned audio, and record whether a local stretch happened.
    yield module
    module.synthesize_request = _ORIGINAL_SYNTHESIZE


_ORIGINAL_SYNTHESIZE = None


def _install_fakes(monkeypatch, tts_core, calls):
    global _ORIGINAL_SYNTHESIZE
    _ORIGINAL_SYNTHESIZE = tts_core.synthesize_request

    def fake_synthesize(text, *, model, voice, speed, base_url, key, instructions="", response_format="mp3"):
        calls["speed_arg"] = speed
        calls["model"] = model
        calls["voice_arg"] = voice
        calls["response_format"] = response_format
        return b"ID3fake-audio-bytes"

    def fake_stretch(path, rate):
        calls["stretched_with"] = rate
        return path

    monkeypatch.setattr(tts_core, "synthesize_request", fake_synthesize)
    monkeypatch.setattr(tts_core, "time_stretch", fake_stretch)
    monkeypatch.setattr(tts_core, "resolve_tts_settings",
                        lambda explicit="": {"model": "deepgram/aura-2", "voice": "aura-2-thalia-en",
                                             "base_url": "https://x/v1", "speed": 1.0, "api_key": "k"})


def test_atempo_chain_covers_the_whole_range(common):
    assert common.build_atempo_chain(1.0) == ""
    assert common.build_atempo_chain(1.5) == "atempo=1.5000"
    assert common.build_atempo_chain(2.0) == "atempo=2.0000"
    # one atempo instance only covers 0.5-2.0, so out-of-range rates chain
    assert common.build_atempo_chain(3.0) == "atempo=2.0,atempo=1.5000"
    assert common.build_atempo_chain(0.4) == "atempo=0.5,atempo=0.8000"
    # below the range: 0.5 x 0.5 = 0.25 (the documented floor)
    assert common.build_atempo_chain(0.1) == "atempo=0.5,atempo=0.5000"


def test_honoured_model_sends_speed_natively(tts_core, monkeypatch, tmp_path):
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "a.mp3"), model="deepgram/aura-2", speed=1.5)

    assert calls["speed_arg"] == 1.5, "a model measured to honour speed should receive it"
    assert "stretched_with" not in calls, "must not double-apply speed natively AND locally"


def test_ignoring_model_gets_local_stretch_instead(tts_core, monkeypatch, tmp_path):
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "b.mp3"), model="x-ai/grok-voice-tts-1.0", speed=1.5)

    assert calls["speed_arg"] is None, "a model that ignores speed must not be sent the parameter"
    assert calls["stretched_with"] == 1.5, "the rate must still be applied locally"


def test_qwen_is_never_sent_speed(tts_core, monkeypatch, tmp_path):
    """It answers HTTP 400 'does not support the speed parameter' — sending it breaks the call."""
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "c.mp3"),
                                model="qwen/qwen-audio-3.0-tts-flash", speed=1.5)

    assert calls["speed_arg"] is None
    assert calls["stretched_with"] == 1.5


def test_speed_one_is_a_no_op(tts_core, monkeypatch, tmp_path):
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "d.mp3"), model="deepgram/aura-2", speed=1.0)

    assert calls["speed_arg"] is None
    assert "stretched_with" not in calls


def test_configured_pcm_response_is_wrapped_as_wav(tts_core, monkeypatch, tmp_path):
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)
    monkeypatch.setattr(tts_core, "resolve_tts_settings", lambda explicit="": {
        "model": "google/gemini-3.1-flash-tts-preview",
        "voice": "Zephyr",
        "base_url": "https://x/v1",
        "speed": 1.0,
        "file_format": "pcm",
        "api_key": "k",
    })
    monkeypatch.setattr(tts_core, "synthesize_request", lambda *args, **kwargs: (
        calls.update(response_format=kwargs["response_format"]) or b"\x00\x00" * 24
    ))

    result = tts_core.synthesize_to_file("hello", str(tmp_path / "reply.mp3"))

    assert calls["response_format"] == "pcm"
    assert result.endswith("reply.wav")
    with wave.open(result, "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.readframes(audio.getnframes()) == b"\x00\x00" * 24


def test_speed_is_clamped_to_the_tool_range(tts_core, monkeypatch, tmp_path):
    """9.0 clamps to 4.0; kokoro's measured range ends at 3.0, so 3.0 native + 1.333 local."""
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "e.mp3"), model="hexgrad/kokoro-82m", speed=9.0)

    assert calls["speed_arg"] == 3.0
    assert calls["stretched_with"] == pytest.approx(4.0 / 3.0)


def test_rate_outside_the_models_range_is_split(tts_core, monkeypatch, tmp_path):
    """aura-2 rejects 1.6 with a 400, so 1.5 goes to the API and 1.0667 is stretched locally."""
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "f.mp3"), model="deepgram/aura-2", speed=1.6)

    assert calls["speed_arg"] == pytest.approx(1.5)
    assert calls["stretched_with"] == pytest.approx(1.6 / 1.5)


def test_slow_rates_below_the_range_are_split_too(tts_core, monkeypatch, tmp_path):
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "g.mp3"), model="deepgram/aura-2", speed=0.5)

    assert calls["speed_arg"] == pytest.approx(0.7)
    assert calls["stretched_with"] == pytest.approx(0.5 / 0.7)


def test_model_with_no_published_voices_omits_the_voice(tts_core, monkeypatch, tmp_path):
    """Filling in another model's default voice would be wrong, and an empty string is a schema error."""
    calls: dict = {}
    _install_fakes(monkeypatch, tts_core, calls)

    tts_core.synthesize_to_file("hi", str(tmp_path / "h.mp3"), model="fish-audio/s1")
    assert calls["voice_arg"] == ""

    tts_core.synthesize_to_file("hi", str(tmp_path / "i.mp3"), model="deepgram/aura-2")
    assert calls["voice_arg"] == "aura-2-thalia-en"


def test_split_speed_is_total(common):
    assert common.split_speed("deepgram/aura-2", 1.0) == (None, 1.0)
    assert common.split_speed("deepgram/aura-2", 1.5) == (1.5, 1.0)
    assert common.split_speed("x-ai/grok-voice-tts-1.0", 1.5) == (None, 1.5)   # no measured range
    native, local = common.split_speed("deepgram/aura-2", 2.0)
    assert native == 1.5 and abs(native * local - 2.0) < 1e-9
