"""Embed each row of english_senses.jsonl with Gemma's hidden state at a chosen inner layer.

For each input "{word}: {gloss}", we run one forward pass and capture the
hidden state at the OUTPUT of `model.model.language_model.layers[layer_idx]`
via a forward hook, then take the LAST-token slice. Default layer_frac=2/3
lands late enough to be semantically abstract, early enough to dodge the
final layer's specialization for next-token prediction.

Speed setup
-----------
- `attn_implementation="sdpa"`: FA2/FA3 reject Gemma 4's head_dim=512 (their max is 256).
  See: https://github.com/Dao-AILab/flash-attention/issues/2427
- `torch.compile(model.forward, mode="reduce-overhead")`: v5's supported
  compile path. We don't pass `fullgraph=True` because transformers'
  attention-mask helper has a data-dependent branch that would reject it.
- Every batch is left-padded to a fixed `--max-length` so the compiled
  graph reuses one shape and we don't recompile mid-run.
- A forward hook captures the chosen layer's output instead of
  `output_hidden_states=True`, which would force a graph break.

Output (default `data/`)
------------------------
    embeddings.bin   fp16 [N, hidden_dim] memmap; row i ↔ input row i
    meta.json        run config + shape
    progress.txt     last completed row index (for crash-resume)
"""

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

torch.set_float32_matmul_precision("high")

MODEL_ID = "google/gemma-4-31B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("data/english_senses.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--layer-frac", type=float, default=2 / 3,
                        help="Layer index = round(layer_frac * num_hidden_layers)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128,
                        help="Every batch is left-padded/truncated to exactly this many tokens")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (useful while debugging)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N input rows")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.out_dir / "embeddings.bin"
    meta_path = args.out_dir / "meta.json"
    progress_path = args.out_dir / "progress.txt"

    n_rows = sum(1 for _ in args.rows.open())
    if args.limit is not None:
        n_rows = min(n_rows, args.limit)

    processor = AutoProcessor.from_pretrained(args.model_id)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    # Gemma 4 is multimodal: text decoder lives at model.model.language_model.layers;
    # hidden_size sits inside config.text_config (Gemma4Config is a composite).
    text_layers = model.model.language_model.layers
    n_layers = len(text_layers)
    hidden_dim = model.config.text_config.hidden_size
    layer_idx = round(args.layer_frac * n_layers)
    print(
        f"model={args.model_id} n_layers={n_layers} hidden_dim={hidden_dim} "
        f"-> layer_idx={layer_idx}",
        file=sys.stderr,
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    text_layers[layer_idx].register_forward_hook(hook)

    if not args.no_compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    start = 0
    if progress_path.exists() and emb_path.exists():
        start = int(progress_path.read_text().strip() or 0)
    if start >= n_rows:
        print(f"already complete ({start:,} rows)", file=sys.stderr)
        return

    mode = "r+" if start > 0 else "w+"
    embeddings = np.memmap(
        emb_path, dtype=np.float16, mode=mode, shape=(n_rows, hidden_dim),
    )
    meta_path.write_text(json.dumps({
        "n": n_rows,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
        "layer_idx": layer_idx,
        "n_layers": n_layers,
        "layer_frac": args.layer_frac,
        "model_id": args.model_id,
        "attn_implementation": "sdpa",
        "rows_path": str(args.rows),
        "max_length": args.max_length,
        "compiled": not args.no_compile,
    }, indent=2))
    print(f"writing rows [{start:,}, {n_rows:,}) to {emb_path}", file=sys.stderr)

    @torch.inference_mode()
    def embed(texts: list[str]) -> np.ndarray:
        toks = processor(
            text=texts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=args.max_length,
        ).to(model.device)
        model(**toks, use_cache=False)
        h = captured["h"]  # [B, T, D] — populated by the forward hook
        return h[:, -1, :].to(torch.float16).cpu().numpy()  # left-pad → last is real

    written = start
    t0 = time.time()
    batch: list[str] = []

    def flush() -> None:
        nonlocal written
        if not batch:
            return
        emb = embed(batch)
        embeddings[written:written + emb.shape[0]] = emb
        written += emb.shape[0]
        embeddings.flush()
        progress_path.write_text(str(written))
        batch.clear()

    with args.rows.open() as f:
        for line in islice(f, start, n_rows):
            batch.append(json.loads(line)["text"])
            if len(batch) >= args.batch_size:
                flush()
                if (written // args.batch_size) % 50 == 0:
                    rate = (written - start) / max(time.time() - t0, 1e-9)
                    eta_min = (n_rows - written) / max(rate, 1e-9) / 60
                    print(
                        f"written={written:,}/{n_rows:,} "
                        f"({rate:.1f} rows/s, eta {eta_min:.1f}m)",
                        file=sys.stderr,
                    )
        flush()

    print(f"done: {written:,} embeddings -> {emb_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
