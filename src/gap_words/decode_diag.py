"""Diagnostic for decode pipeline. Three sanity checks:

1. Self round-trip: re-run PromptEOL on a known row, confirm captured h matches
   stored emb[i]. Tests the embedding pipeline itself.
2. Same-prompt patch: patch h_i back into the SAME row's prompt at trailing `"`
   layer 40. Read logits. Should predict word_i's first BPE token. (This is
   redundant with embedding capture but verifies the patch mechanism.)
3. Cross-prompt patch: patch h_A into row B's PromptEOL prompt at trailing `"`,
   layer 40. Read logits. Should predict word_A (concept-transplant test).

This isolates Patchscopes-prompt issues from capture/patch issues.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-31B"
LAYER = 40
PROMPT_TEMPLATE = '"{word}: {gloss}" can be summarized in one word as: "'


def load_embeddings(path: Path, n: int, hidden: int) -> torch.Tensor:
    return torch.from_file(
        str(path), shared=False, size=n * hidden, dtype=torch.bfloat16,
    ).view(n, hidden)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--rows-path", type=Path, default=Path("data/english_senses.jsonl"))
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text())
    n, hidden = meta["n"], meta["hidden_dim"]

    print(f"loading model {MODEL_ID}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", truncation_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    text_layers = model.model.language_model.layers

    emb = load_embeddings(args.data_dir / "embeddings.bin", n, hidden)

    random.seed(args.seed)
    chosen = sorted(random.sample(range(n), args.n_pairs * 2))
    rows: dict[int, dict] = {}
    with args.rows_path.open() as f:
        for i, line in enumerate(f):
            if not chosen:
                break
            if i == chosen[0]:
                rows[i] = json.loads(line)
                chosen.pop(0)
    chosen_idxs = sorted(rows.keys())

    # Capture hook
    captured = {}
    def cap(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured["h"] = t.detach().clone()
    cap_h = text_layers[LAYER].register_forward_hook(cap)

    # Patch hook
    patch_state = {"h": None, "pos": None}
    def patch(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured["h"] = t.detach().clone()  # also capture for reading
        if patch_state["h"] is None:
            return None
        new_t = t.clone()
        new_t[0, patch_state["pos"], :] = patch_state["h"].to(new_t.dtype)
        if isinstance(o, tuple):
            return (new_t,) + o[1:]
        return new_t

    # Replace cap with patch
    cap_h.remove()
    patch_h_handle = text_layers[LAYER].register_forward_hook(patch)

    @torch.inference_mode()
    def run_prompt(prompt: str, patch_h=None, patch_pos: int | None = None):
        if patch_h is not None:
            patch_state["h"] = patch_h.to(model.device, dtype=torch.bfloat16)
            patch_state["pos"] = patch_pos
        else:
            patch_state["h"] = None
            patch_state["pos"] = None
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -1]
        return ids[0], captured["h"][0], logits

    print("\n=== TEST 1: self round-trip (re-run PromptEOL prompt for row i, check captured h matches stored emb[i]) ===")
    for idx in chosen_idxs[:args.n_pairs]:
        row = rows[idx]
        prompt = PROMPT_TEMPLATE.format(word=row["word"], gloss=row["gloss"])
        ids, h_captured, logits = run_prompt(prompt)
        h_stored = emb[idx].to(model.device, dtype=torch.bfloat16)
        h_re = h_captured[-1].to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(
            h_re.unsqueeze(0), h_stored.to(torch.float32).unsqueeze(0)
        ).item()
        l2 = (h_re - h_stored.to(torch.float32)).norm().item()
        # Also: top-1 logit at this position
        top = logits.softmax(dim=-1).topk(3)
        top_strs = ", ".join(f"{tok.decode([int(t)]).strip()!r}({float(p):.2f})"
                             for t, p in zip(top.indices.tolist(), top.values.tolist()))
        print(f"  idx={idx} word={row['word']!r:30s}  cos(stored,recap)={cos:.5f}  L2_diff={l2:.4f}  natural_top1: {top_strs}")

    print("\n=== TEST 2: same-prompt patch (patch h_i back into row i's prompt at trailing `\"`) ===")
    # Should give same logits as TEST 1 (no real perturbation since h is what's already there)
    for idx in chosen_idxs[:args.n_pairs]:
        row = rows[idx]
        prompt = PROMPT_TEMPLATE.format(word=row["word"], gloss=row["gloss"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        last_pos = ids.shape[1] - 1
        h = emb[idx].to(torch.float32)
        _, _, logits = run_prompt(prompt, patch_h=h, patch_pos=last_pos)
        top = logits.softmax(dim=-1).topk(3)
        top_strs = ", ".join(f"{tok.decode([int(t)]).strip()!r}({float(p):.2f})"
                             for t, p in zip(top.indices.tolist(), top.values.tolist()))
        print(f"  idx={idx} word={row['word']!r:30s}  patched_top1: {top_strs}")

    print("\n=== TEST 3: cross-prompt patch (patch h_A into row B's PromptEOL prompt) ===")
    # If patch dominates, model should output word_A (the patched concept) regardless of B's context
    pairs = [(chosen_idxs[i], chosen_idxs[i + args.n_pairs]) for i in range(args.n_pairs)]
    for ia, ib in pairs:
        row_a, row_b = rows[ia], rows[ib]
        prompt_b = PROMPT_TEMPLATE.format(word=row_b["word"], gloss=row_b["gloss"])
        ids_b = tok(prompt_b, return_tensors="pt").input_ids.to(model.device)
        last_pos = ids_b.shape[1] - 1
        h_a = emb[ia].to(torch.float32)
        _, _, logits = run_prompt(prompt_b, patch_h=h_a, patch_pos=last_pos)
        top = logits.softmax(dim=-1).topk(3)
        top_strs = ", ".join(f"{tok.decode([int(t)]).strip()!r}({float(p):.2f})"
                             for t, p in zip(top.indices.tolist(), top.values.tolist()))
        print(f"  patch={row_a['word']!r:25s} into prompt-of={row_b['word']!r:25s} -> {top_strs}")

    # ----- TEST 4: minimal-prior-context prompts. Patch at trailing `"`. -----
    minimal_prompts = [
        # Variant 1: empty word, empty gloss
        '": " can be summarized in one word as: "',
        # Variant 2: generic word/gloss
        '"thing: a concept." can be summarized in one word as: "',
        # Variant 3: just the suffix
        '" can be summarized in one word as: "',
        # Variant 4: brand-new minimal prompt
        '"',
    ]
    print("\n=== TEST 4: minimal-prior-context prompts, patch at trailing `\"` ===")
    test_idxs = chosen_idxs[:args.n_pairs]
    for variant_idx, p_template in enumerate(minimal_prompts):
        ids = tok(p_template, return_tensors="pt").input_ids.to(model.device)
        last_pos = ids.shape[1] - 1
        # Control (no patch)
        _, _, logits_c = run_prompt(p_template)
        top_c = logits_c.softmax(dim=-1).topk(3)
        top_c_strs = ", ".join(f"{tok.decode([int(t)]).strip()!r}({float(p):.2f})"
                               for t, p in zip(top_c.indices.tolist(), top_c.values.tolist()))
        print(f"\n  variant {variant_idx}: prompt={p_template!r}  ({ids.shape[1]} tokens)")
        print(f"    control (no patch): {top_c_strs}")
        for idx in test_idxs:
            row = rows[idx]
            h = emb[idx].to(torch.float32)
            _, _, logits = run_prompt(p_template, patch_h=h, patch_pos=last_pos)
            top = logits.softmax(dim=-1).topk(3)
            top_strs = ", ".join(f"{tok.decode([int(t)]).strip()!r}({float(p):.2f})"
                                 for t, p in zip(top.indices.tolist(), top.values.tolist()))
            print(f"    patch={row['word']!r:30s} -> {top_strs}")


if __name__ == "__main__":
    main()
