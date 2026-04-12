# BlasBench

An open benchmark for Irish (Gaeilge) speech recognition.

BlasBench evaluates automatic speech recognition systems on Irish with a normaliser
that preserves what Irish actually is: fadas (á é í ó ú), lenition, and eclipsis are
kept intact rather than stripped to ASCII. It reports Word Error Rate and Character
Error Rate on Common Voice ga-IE and FLEURS ga-IE, covering 12 models across four
architecture families (Whisper, SeamlessM4T, wav2vec2/MMS, commercial APIs). If the
eval is wrong, everything built on top is wrong — this is the ground truth layer.

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
`@model_registry.register("your-model-name")`. There currently also three commercial adapters
(`azure_adapter.py`, `openai_adapter.py`, `elevenlabs_adapter.py`) to support 
commercial models. 

## Citation

The paper describing BlasBench and the 12-model evaluation will be posted to
arXiv shortly. A `CITATION.cff` will be added at that point.

## License

MIT. See `LICENSE`.
