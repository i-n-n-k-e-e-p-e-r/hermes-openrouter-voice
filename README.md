# hermes-openrouter-voice

Adds **OpenRouter** to both voice directions in [Hermes Agent](https://github.com/NousResearch/hermes-agent):

| Direction | Config | Endpoint |
|---|---|---|
| Speech-to-text | `stt.provider: openrouter` | `POST /api/v1/audio/transcriptions` |
| Text-to-speech | `tts.provider: openrouter` | `POST /api/v1/audio/speech` |

Both reuse **`OPENROUTER_API_KEY`** — the key your chat provider already uses — so there is no second
account or bill. Stdlib only, no pip dependencies.

Upstream closed the "built-in OpenRouter STT provider" request as not-planned
([#24415](https://github.com/NousResearch/hermes-agent/issues/24415)) in favour of the provider surfaces
(plugins and `stt.providers.<name>` command providers). This plugin is that path, packaged.

## Install

```bash
hermes plugins install esketids/hermes-openrouter-voice
hermes plugins enable openrouter-voice
```

Per profile (each profile has its own config and plugins), add `--profile <name>` to both commands.
Later: `hermes plugins update openrouter-voice`.

## Configure

```bash
hermes config set stt.provider openrouter
hermes config set tts.provider openrouter      # only if you want OpenRouter speech output
```

Optional model choices (defaults shown):

```yaml
stt:
  openrouter:
    model: openai/whisper-large-v3     # or meta/muse-voice-transcribe-1.0, google/chirp-3, …
    language: ""                       # "" = auto-detect
tts:
  openrouter:
    model: deepgram/aura-2
    voice: aura-2-thalia-en            # voices are MODEL-SPECIFIC
    speed: 1.0                         # 0.25-4.0
```

## In the desktop GUI

**Without any patch:** the two provider *names* become selectable in Settings → Voice (the model and
voice rows do not appear). Hermes builds that dropdown from a bundle-compiled list plus any **command
provider declared in config**, so declare these two:

```bash
PY=~/.hermes/hermes-agent/venv/bin/python
PLUG=~/.hermes/plugins/openrouter-voice

hermes config set --force stt.providers.openrouter.type command
hermes config set --force stt.providers.openrouter.command \
  "$PY $PLUG/transcribe.py --config ~/.hermes/config.yaml {input_path} {output_path} {language} {model}"
hermes config set --force tts.providers.openrouter.type command
hermes config set --force tts.providers.openrouter.command \
  "$PY $PLUG/speak.py --config ~/.hermes/config.yaml {input_path} {output_path} {voice} {model}"
hermes config set --force tts.providers.openrouter.voice_compatible true
```

`--force` is required (without it Hermes writes nothing for these dotted registry keys), and
`--config <path>` is not optional: profiles are not always distinguished by `HERMES_HOME`, so the shim
must be told which config to read.

**With the GUI patch:** adds the OpenRouter **model / voice / speed rows**, a **Preview** button beside
the voice and speed rows (speaks a sample using the settings in force at that moment), makes the voice
list **follow the selected model**, and adds the **microphone / speaker device pickers**.

The rows live in a bundle-compiled list (`SECTIONS` in `apps/desktop/src/app/settings/constants.ts`), so
no plugin or config entry can add one — a plugin contributes Python providers, not UI. The patch is kept
**out of this repo's tree** (a catalog entry may not ship a patch for the app) and lives on branch
**`gui-rows`**, with a copy on each machine that uses it:

```bash
~/.hermes/patches/openrouter-voice-gui/apply-gui-rows.sh               # patch + repack (~5 min)
~/.hermes/patches/openrouter-voice-gui/apply-gui-rows.sh --no-pack     # patch only
```

Three things to know, because they are what make it feel fragile:

* `hermes update` replaces the checkout and rebuilds the app, so the rows disappear — re-run the script.
  A **watchdog** (`gui/watch-gui-rows.sh`, run by cron) notices, re-applies what it can, and speaks only
  when something needs a human; it stays silent while healthy.
* The patch is cut against **one commit**, and upstream edits these files most releases, so after an
  update it fails **loudly** (naming its base and your HEAD) instead of half-applying. Re-cut it.
* The device pickers need the backend to **restart**, not just the app to reload: their keys come from
  the schema, which the Python process reads at boot.

The pickers use **browser device ids** (`navigator.mediaDevices.enumerateDevices()`), a different
namespace from `wake_word.input_device` — that one is a **PortAudio** index for wake-word capture on the
Python side. A chosen device that is unplugged raises `OverconstrainedError`: recording falls back to the
system default and the row says so, rather than switching silently.

## What the endpoint actually accepts (measured, not assumed)

| Finding | Detail |
|---|---|
| `response_format` | transcription accepts only `json` / `verbose_json`; `"text"` is a **400** |
| Sample rate | `meta/muse-voice-transcribe-1.0` requires **16 or 24 kHz mono WAV**; 22.05 kHz is a live 400. PCM WAV is normalised in pure Python, no ffmpeg |
| Containers | every model but that one accepts MP3/WebM; the desktop records `audio/webm;codecs=opus`, which needs **ffmpeg** to reach a WAV-only model |
| Truncation | `hexgrad/kokoro-82m` returns **only the first sentence** — not the default |
| Format | `google/gemini-3.1-flash-tts-preview` requires `pcm`; configure `tts.openrouter.file_format: pcm`. The raw 24 kHz PCM is wrapped as WAV for playback |
| Voices | model-specific: `aura-2-*` for `deepgram/aura-2`, `af_heart` for Kokoro |
| Verified default | `deepgram/aura-2` + `aura-2-thalia-en` — speaks multi-sentence input in full |

## Models, voices, speed

```bash
python models.py                     # shipped catalogs (no network)
python models.py --voices [model]    # a model's voice set
python models.py --live              # diff the shipped lists against the API
```

| Config key | Shipped | Source |
|---|---|---|
| `stt.openrouter.model` | **21** transcription models | `GET /api/v1/models?output_modalities=transcription` |
| `tts.openrouter.model` | **18** speech models | `GET /api/v1/models?output_modalities=speech` |
| `tts.openrouter.voice` | **361** voices across 16 models | each model's `supported_voices` (there is no voices endpoint) |
| `tts.openrouter.file_format` | `mp3` by default; `pcm` for Gemini TTS | OpenRouter speech response format; PCM is wrapped as 24 kHz mono WAV |

For Gemini TTS, configure the model's required response format explicitly:

```yaml
tts:
  openrouter:
    model: google/gemini-3.1-flash-tts-preview
    voice: Zephyr
    file_format: pcm
```

Voice counts: `deepgram/aura-2` 90, `hexgrad/kokoro-82m` 54, `minimax/speech-2.8-*` 45,
`mistralai/voxtral-mini-tts-2603` 30, `microsoft/mai-voice-2` 4, `fish-audio/*` **none published**
(leave the voice empty; the provider uses its default).

`tts.openrouter.speed` is honoured on every model, two ways: sent **natively** where the model supports it
(`aura-2` accepts only 0.7-1.5 — 1.6 is a 400; `qwen/*` rejects the parameter entirely), otherwise applied
as a local pitch-preserving time-stretch (`ffmpeg atempo`). Measured with `speed=1.5`: audio landed at
0.60-0.68 of baseline; expect ±10-20 %, since providers pace themselves differently between runs. Without
ffmpeg the rate is sent to the API only.

Voice samples (the API publishes no preview URLs):

```bash
python models.py --sample deepgram/aura-2              # one clip per voice, cached
python models.py --sample deepgram/aura-2 --montage    # plus a concatenated file
```

Each clip names its own voice first, so a montage is self-labelling. Output lands in
`~/.hermes/cache/openrouter-voice-samples/<model>/`.

## Verify / layout

```bash
$PY ~/.hermes/plugins/openrouter-voice/transcribe.py --config ~/.hermes/config.yaml clip.wav
python -m pytest tests/ -q          # audio normaliser: downmix, resample, idempotence
```

```
__init__.py      register(ctx) → registers BOTH providers
common.py        credentials, profile-aware config, WAV normalise/resample, HTTP
stt_core.py      transcription logic (no Hermes imports)     stt.py   provider wrapper
tts_core.py      synthesis logic (no Hermes imports)         tts.py   provider wrapper
transcribe.py / speak.py   command-provider shims (GUI selectability)
models.py        catalogs, voices, samples     tests/   stdlib unit tests
```

The `*_core.py` modules import nothing from Hermes — the shims run outside the repo's `sys.path`, so the
ABC subclasses live in `stt.py` / `tts.py` and the logic is shared.

## Related upstream work

* **Listed in the Hermes plugin catalog** — catalog entry merged 2026-09-20.
* [#24415](https://github.com/NousResearch/hermes-agent/issues/24415) — STT provider (closed, not-planned; this plugin is the sanctioned alternative)
* [#15726](https://github.com/NousResearch/hermes-agent/issues/15726) — TTS provider (open)
* [#117088](https://github.com/NousResearch/hermes-agent/pull/117088) — microphone/speaker pickers for everyone (open)
* [#112122](https://github.com/NousResearch/hermes-agent/pull/112122) / [#112126](https://github.com/NousResearch/hermes-agent/pull/112126) — built-in provider variants (open; upstream prefers extending the existing transports instead)

## Licence

MIT
