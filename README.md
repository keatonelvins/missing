# missing

Find concepts in an LM's latent representation that have no English word.

Two scripts — run in order:

## 1. Build the English sense dataset

Streams the [kaikki.org](https://kaikki.org/) wiktextract dump and emits one JSONL row per English leaf sense.

```bash
uv run python -m missing.build_dataset
# -> data/english_senses.jsonl  (≈1.75M rows from a 10.6M-entry dump, ~4 min)
```

Each row: `{word, pos, sense_idx, tags, topics, categories, gloss, gloss_path, text}`. The `text` field is `"{word}: {gloss}"` — what we feed the LM.

Smoke test: `--limit 100 --out data/smoke.jsonl`.

## 2. Embed each sense with Gemma-4-31B

For each row, runs one forward pass and captures the hidden state at the output of layer `round(2/3 * num_hidden_layers) = 40` of 60 (last token, fp16).

```bash
uv run python -m missing.embed
# -> data/embeddings.bin   fp16 memmap [N, 5376]
#    data/meta.json        run config
#    data/progress.txt     resume marker
```

Crash-resume: rerun the same command. Smoke test: `--limit 200 --out-dir data/smoke`.

Knobs: `--layer-frac`, `--batch-size`, `--max-length`, `--no-compile`, `--model-id`.
