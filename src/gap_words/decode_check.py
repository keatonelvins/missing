"""Patchscopes round-trip sanity check on known sense embeddings.

Stage 0.2 + 0.3 of the gap-words plan.

For each of N sampled rows from english_senses.jsonl:
  - Load the stored layer-40 embedding h_orig.
  - Patch h_orig into the residual stream at the `?` position of a
    few-shot identity prompt at layer 40, run forward, take top-k logits at
    the final position. If top-1 ≈ row's word, decoding works.
  - (Optional) Re-embed top-1 token through the original PromptEOL pipeline
    and report cos(h_orig, h_reembedded) — the NLA-style round-trip.

We're not training anything. Just frozen base model + a hook.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-31B"
LAYER = 40
PROMPT_TEMPLATE = '"{word}: {gloss}" can be summarized in one word as: "'

# Decoder prompt. `decode_diag.py` showed that the Patchscopes few-shot
# identity prompt fails for our setup: the literal placeholder token's
# pre-layer-40 representation poisons the patched representation through
# attention from later layers. A minimal prompt with no competing context
# decodes well — the patch becomes the only signal at the answer position.
# We patch h at the trailing `"` (the LAST position in the prompt).
PATCHSCOPES_PROMPT = 'can be summarized in one word as: "'


def load_embeddings(path: Path, n: int, hidden: int) -> torch.Tensor:
    return torch.from_file(
        str(path), shared=False, size=n * hidden, dtype=torch.bfloat16,
    ).view(n, hidden)


def tokenize_and_patch_position(tokenizer, prompt: str) -> tuple[torch.Tensor, int]:
    """Tokenize prompt; the patch goes at the LAST position (trailing `\"`)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return ids, ids.shape[0] - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--rows-path", type=Path, default=Path("data/english_senses.jsonl"))
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("data/decode_check.json"))
    ap.add_argument("--no-roundtrip", action="store_true",
                    help="Skip the NLA-style re-embedding round-trip cosine.")
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text())
    n, hidden = meta["n"], meta["hidden_dim"]
    print(f"loading model {MODEL_ID}", file=sys.stderr)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, padding_side="left", truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    text_layers = model.model.language_model.layers

    # Resolve patch position once (last position of the prompt — the trailing `"`).
    patch_ids, patch_pos = tokenize_and_patch_position(tok, PATCHSCOPES_PROMPT)
    patch_ids = patch_ids.unsqueeze(0).to(model.device)  # [1, T]
    print(f"decoder prompt has {patch_ids.shape[1]} tokens; patching position {patch_pos} "
          f"(token id {patch_ids[0, patch_pos].item()} = {tok.decode(patch_ids[0, patch_pos:patch_pos+1])!r})",
          file=sys.stderr)

    # Hook holder — set per-row before each forward
    state = {"h": None, "fired": 0, "shape": None, "is_tuple": None}

    def hook(_mod, _inp, output):
        state["fired"] += 1
        is_tuple = isinstance(output, tuple)
        state["is_tuple"] = is_tuple
        t = output[0] if is_tuple else output
        state["shape"] = tuple(t.shape)
        if state["h"] is None:
            return None  # no-op
        new_t = t.clone()
        new_t[:, patch_pos, :] = state["h"].to(new_t.dtype).to(new_t.device)
        if is_tuple:
            return (new_t,) + output[1:]
        return new_t

    text_layers[LAYER].register_forward_hook(hook)

    # Load full embeddings (bf16, mmap)
    emb_path = args.data_dir / "embeddings.bin"
    emb = load_embeddings(emb_path, n, hidden)

    # Sample rows
    random.seed(args.seed)
    chosen = sorted(random.sample(range(n), args.n_samples))
    by_idx: dict[int, dict] = {}
    with args.rows_path.open() as f:
        for i, line in enumerate(f):
            if not chosen:
                break
            if i == chosen[0]:
                by_idx[i] = json.loads(line)
                chosen.pop(0)
    chosen_idxs = sorted(by_idx.keys())

    # ----- Patchscopes decode -----
    @torch.inference_mode()
    def patchscopes_decode(
        h: torch.Tensor | None, top_k: int = 5, scale: float | None = None,
    ) -> list[tuple[int, float, str]]:
        """Decode top-k tokens at the final position with `h` patched in.

        scale=None: patch h as-is.
        scale=<float>: rescale h to that L2 norm before patching (useful for
        matching the natural norm at the placeholder position).
        h=None: control, no patching.
        """
        if h is None:
            state["h"] = None
        else:
            h_dev = h.to(model.device, dtype=torch.float32)
            if scale is not None:
                h_dev = h_dev * (scale / h_dev.norm().clamp_min(1e-9))
            state["h"] = h_dev.to(torch.bfloat16)
        out = model(input_ids=patch_ids, use_cache=False)
        state["h"] = None
        logits = out.logits[0, -1]
        probs = logits.softmax(dim=-1)
        top = probs.topk(top_k)
        return [
            (int(tid), float(p), tok.decode([int(tid)]))
            for tid, p in zip(top.indices.tolist(), top.values.tolist())
        ]

    # ----- Re-embedding for round-trip cosine -----
    @torch.inference_mode()
    def reembed_word(word: str) -> torch.Tensor:
        """Run a degenerate PromptEOL pass for a single word with no gloss
        (so we measure 'embed of just the word' through the same prompt frame)."""
        captured = {}
        def cap_hook(_m, _i, o):
            captured["h"] = o[0] if isinstance(o, tuple) else o
        h_handle = text_layers[LAYER].register_forward_hook(cap_hook)
        try:
            text = PROMPT_TEMPLATE.format(word=word, gloss=word)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            model(input_ids=ids)
            return captured["h"][0, -1, :].to("cpu", dtype=torch.float32).clone()
        finally:
            h_handle.remove()

    # ----- Diagnostic 1: control no-patch decode -----
    print("\n--- control: no-patch decode of Patchscopes prompt ---", file=sys.stderr)
    ctrl = patchscopes_decode(None, top_k=args.top_k)
    print(f"  hook fired={state['fired']} is_tuple={state['is_tuple']} shape={state['shape']}", file=sys.stderr)
    for tid, p, s in ctrl:
        print(f"    {p:.3f}  {s!r}", file=sys.stderr)

    # ----- Diagnostic 2: natural h at placeholder position (for scale) -----
    captured_natural = {}
    def cap_natural(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured_natural["h"] = t[0, patch_pos, :].clone()
    h_handle = text_layers[LAYER].register_forward_hook(cap_natural)
    with torch.inference_mode():
        model(input_ids=patch_ids, use_cache=False)
    h_handle.remove()
    natural_norm = captured_natural["h"].float().norm().item()
    source_norms = emb[chosen_idxs[:5]].float().norm(dim=1).tolist()
    print(f"\n--- diagnostic norms ---", file=sys.stderr)
    print(f"  natural h@layer{LAYER}@pos{patch_pos} norm in Patchscopes prompt: {natural_norm:.2f}", file=sys.stderr)
    print(f"  sample source-h norms (first 5 rows): {[f'{n:.2f}' for n in source_norms]}", file=sys.stderr)

    # ----- Diagnostic 3: logit lens directly on h (bypass patch) -----
    @torch.inference_mode()
    def logit_lens(h: torch.Tensor, top_k: int = 5) -> list[tuple[int, float, str]]:
        # Apply final norm + lm_head to h.
        # Gemma 4 has model.model.language_model.norm and model.lm_head
        h_dev = h.to(model.device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        normed = model.model.language_model.norm(h_dev)
        logits = model.lm_head(normed)[0, 0]
        probs = logits.softmax(dim=-1)
        top = probs.topk(top_k)
        return [
            (int(tid), float(p), tok.decode([int(tid)]))
            for tid, p in zip(top.indices.tolist(), top.values.tolist())
        ]
    print(f"\n--- logit-lens decode of stored h (no patching, just unembed) ---", file=sys.stderr)
    for idx in chosen_idxs[:5]:
        row = by_idx[idx]
        h = emb[idx].to(torch.float32)
        ll = logit_lens(h, top_k=args.top_k)
        ll_strs = ", ".join(f"{s.strip()!r}({p:.2f})" for _, p, s in ll)
        print(f"  {row['word']!r:30s}  -> {ll_strs}", file=sys.stderr)
    print(file=sys.stderr)

    # ----- Diagnostic 4: scan over scale on a few rows -----
    print(f"\n--- scale scan: patch h at norms {{native, natural, 0.5x, 2x}} ---", file=sys.stderr)
    scale_options = [
        ("native", None),
        ("natural", natural_norm),
        ("half_natural", natural_norm * 0.5),
        ("double_natural", natural_norm * 2.0),
    ]
    for idx in chosen_idxs[:3]:
        row = by_idx[idx]
        h = emb[idx].to(torch.float32)
        print(f"  word={row['word']!r}  source_norm={h.norm().item():.2f}", file=sys.stderr)
        for label, sc in scale_options:
            top = patchscopes_decode(h, top_k=3, scale=sc)
            top_strs = ", ".join(f"{s.strip()!r}({p:.2f})" for _, p, s in top)
            sc_str = f"{sc:.1f}" if sc is not None else "—"
            print(f"    scale={label:14s} ({sc_str:>5s}): {top_strs}", file=sys.stderr)
    print(file=sys.stderr)

    # ----- Run main loop with scale=natural (the NLA recipe) -----
    print(f"\n--- main loop: scale to natural placeholder norm = {natural_norm:.2f} ---", file=sys.stderr)
    results = []
    n_top1_match = 0
    n_topk_match = 0
    for idx in chosen_idxs:
        row = by_idx[idx]
        word = row["word"]
        gloss = row["gloss"]
        h = emb[idx].to(torch.float32)
        top = patchscopes_decode(h, top_k=args.top_k, scale=natural_norm)
        decoded_strs = [t.strip() for _, _, t in top]
        # Match: any of top-k token decodes equals (case-insensitive) word's first whitespace-delimited piece
        word_first = word.split()[0].lower()
        first_subtoken = word.split()[0]
        match_top1 = decoded_strs[0].lower() == word_first
        match_topk = any(s.lower() == word_first for s in decoded_strs)
        # Looser: substring match against word
        match_top1_loose = first_subtoken.lower().startswith(decoded_strs[0].lower()) if decoded_strs[0] else False

        rt_cos = None
        if not args.no_roundtrip and decoded_strs[0]:
            try:
                h_re = reembed_word(decoded_strs[0])
                cos = torch.nn.functional.cosine_similarity(h.unsqueeze(0), h_re.unsqueeze(0)).item()
                rt_cos = cos
            except Exception as e:
                rt_cos = None
                print(f"  reembed failed for {decoded_strs[0]!r}: {e}", file=sys.stderr)

        if match_top1:
            n_top1_match += 1
        if match_topk:
            n_topk_match += 1

        result = {
            "idx": idx,
            "word": word,
            "gloss": gloss[:80] + ("…" if len(gloss) > 80 else ""),
            "top": [(d, round(p, 4)) for _, p, d in top],
            "match_top1": match_top1,
            "match_topk": match_topk,
            "match_top1_loose": match_top1_loose,
            "roundtrip_cos": round(rt_cos, 4) if rt_cos is not None else None,
        }
        results.append(result)
        print(f"[{len(results):>2}/{args.n_samples}] {word!r:35s} -> top1={decoded_strs[0]!r:20s} "
              f"prob={top[0][1]:.3f}  match={match_top1}  rt_cos={rt_cos if rt_cos is None else f'{rt_cos:.3f}'}",
              file=sys.stderr)

    summary = {
        "n_samples": args.n_samples,
        "top1_recovery_rate": n_top1_match / args.n_samples,
        "topk_recovery_rate": n_topk_match / args.n_samples,
        "median_roundtrip_cos": float(torch.tensor(
            [r["roundtrip_cos"] for r in results if r["roundtrip_cos"] is not None]
        ).median()) if not args.no_roundtrip else None,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(file=sys.stderr)
    print(f"top-1 recovery: {n_top1_match}/{args.n_samples} = {summary['top1_recovery_rate']:.0%}", file=sys.stderr)
    print(f"top-{args.top_k} recovery: {n_topk_match}/{args.n_samples} = {summary['topk_recovery_rate']:.0%}", file=sys.stderr)
    if not args.no_roundtrip:
        print(f"median round-trip cos: {summary['median_roundtrip_cos']:.3f}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
