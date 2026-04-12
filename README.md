# BlasBench

An open benchmark for Irish (Gaeilge) speech recognition.

BlasBench evaluates automatic speech recognition systems on Irish with a normaliser
that preserves what Irish actually is: fadas (á é í ó ú), lenition, and eclipsis are
kept intact rather than stripped to ASCII. It reports Word Error Rate and Character
Error Rate on Common Voice ga-IE and FLEURS ga-IE. The accompanying paper evaluates
12 systems spanning four architecture families (Whisper, wav2vec2/MMS, Meta
multilingual, commercial). The harness ships adapters for Whisper and other
transformers models, wav2vec2, MMS, SeamlessM4T, and five commercial APIs (Azure,
OpenAI, ElevenLabs, Speechmatics, Google Cloud). If the eval is wrong, everything
built on top is wrong — this is the ground truth layer.

## Install

```bash
pip install -e .
# or with optional commercial API adapters:
pip install -e ".[api]"
```

## Run a single model

```bash
blasbench evaluate --model whisper-large-v3 --dataset common-voice-ga
```

## Run the full benchmark

```bash
python scripts/run_baselines.py
```

## What you get per run

- `predictions.jsonl` (per-utterance reference, hypothesis, WER, CER)
- aggregate metrics with bootstrap 95% CIs
- S/I/D error breakdown

## Adding a new model

Implement one function: audio (16kHz, `np.ndarray`) in, string out. Wrap it in
a `BaseAdapter` subclass under `src/blasbench/adapters/` and register it via
`@model_registry.register("your-model-name")`. The repo includes five commercial
adapters (`azure_adapter.py`, `openai_adapter.py`, `elevenlabs_adapter.py`,
`speechmatics_adapter.py`, `google_adapter.py`) as working references.

## Reproducing the paper

Tables 1 and 2 (Common Voice ga-IE and FLEURS ga-IE) are produced by the full
baseline sweep plus the figure script:

```bash
python scripts/run_baselines.py           # open-weights models; ~6 H100-hours
python scripts/run_baselines.py --models azure  # needs AZURE_SPEECH_KEY / REGION
```

Per-run artefacts (`predictions.jsonl`, `aggregate.json`) land under
`experiments/`. Commercial APIs are skipped cleanly when env vars are absent.
The omniASR 300M/7B rows in the paper were run from a separate fairseq2
driver and are not yet ported to this harness.

## Citation

The paper describing BlasBench and the 12-model evaluation will be posted to
arXiv shortly. See `CITATION.cff` for a BibTeX-ready entry (the arXiv id is
marked `TODO` until the preprint is live).

## License

MIT. See `LICENSE`.
