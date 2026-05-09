"""Diagnostic statistics on the layer-40 embedding cloud.

Reports:
  - Norm distribution: ‖s‖_2 quantiles. We use the median to set the
    "typical-norm shell" Patchscopes patches will be projected to,
    matching NLA's "unit L2 + fixed scale" input-prep recipe.
  - Random-pair cosine distribution: a baseline for "unrelated".
  - 1-NN cosine distance distribution: what "close to a real word" looks like.
    Stage-1 gap candidates need d_top1 well above this distribution to be
    interesting.

All math runs in fp32 on GPU; embeddings stay bf16 in host memory and are
streamed in chunks.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def load_embeddings(path: Path, n: int, hidden: int) -> torch.Tensor:
    """Open the bf16 mmap as a tensor of shape [n, hidden]."""
    return torch.from_file(
        str(path), shared=False, size=n * hidden, dtype=torch.bfloat16,
    ).view(n, hidden)


def norm_stats(emb: torch.Tensor, sample: int, device: str) -> dict:
    idx = torch.randperm(emb.shape[0])[:sample]
    chunk = emb[idx].to(device, dtype=torch.float32)
    norms = chunk.norm(dim=1)
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
    quants = torch.quantile(norms, qs.to(device)).cpu().tolist()
    return {
        "n_sampled": sample,
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
        "p01": quants[0], "p05": quants[1], "p50": quants[2],
        "p95": quants[3], "p99": quants[4],
    }


def random_pair_cosine(emb: torch.Tensor, n_pairs: int, device: str) -> dict:
    n = emb.shape[0]
    a = torch.randint(0, n, (n_pairs,))
    b = torch.randint(0, n, (n_pairs,))
    # avoid identical-row pairs (extremely rare but cheap to fix)
    same = (a == b)
    b[same] = (b[same] + 1) % n
    A = torch.nn.functional.normalize(emb[a].to(device, dtype=torch.float32), dim=1)
    B = torch.nn.functional.normalize(emb[b].to(device, dtype=torch.float32), dim=1)
    cos = (A * B).sum(dim=1)
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
    quants = torch.quantile(cos, qs.to(device)).cpu().tolist()
    return {
        "n_pairs": n_pairs,
        "cos_mean": cos.mean().item(),
        "cos_std": cos.std().item(),
        "cos_p01": quants[0], "cos_p05": quants[1], "cos_p50": quants[2],
        "cos_p95": quants[3], "cos_p99": quants[4],
    }


def one_nn_distance(
    emb: torch.Tensor, sample: int, device: str, chunk: int = 64,
) -> dict:
    """Sample `sample` rows; find each one's nearest neighbor in the full cloud
    (excluding itself). Report the cosine-distance distribution."""
    n, d = emb.shape
    full_norm = torch.nn.functional.normalize(
        emb.to(device, dtype=torch.float32), dim=1,
    )  # [n, d] on GPU; ~21 GB at fp32 — fine on 80 GB
    idx = torch.randperm(n)[:sample]
    queries = full_norm[idx]  # [sample, d]

    cos_to_nearest = torch.empty(sample, device=device)
    word_idx = torch.arange(n, device=device)
    for start in range(0, sample, chunk):
        end = min(start + chunk, sample)
        q = queries[start:end]  # [b, d]
        scores = q @ full_norm.T  # [b, n]
        # mask self
        rows = idx[start:end].to(device).unsqueeze(1)  # [b, 1]
        scores.scatter_(1, rows, float("-inf"))
        cos_to_nearest[start:end] = scores.max(dim=1).values

    cos_dist = 1.0 - cos_to_nearest
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
    quants = torch.quantile(cos_dist.cpu(), qs).tolist()
    return {
        "n_sampled": sample,
        "cos_dist_mean": cos_dist.mean().item(),
        "cos_dist_std": cos_dist.std().item(),
        "cos_dist_min": cos_dist.min().item(),
        "cos_dist_max": cos_dist.max().item(),
        "cos_dist_p01": quants[0], "cos_dist_p05": quants[1],
        "cos_dist_p50": quants[2], "cos_dist_p95": quants[3],
        "cos_dist_p99": quants[4],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/cloud_stats.json"))
    ap.add_argument("--norm-sample", type=int, default=50_000)
    ap.add_argument("--pair-sample", type=int, default=200_000)
    ap.add_argument("--knn-sample", type=int, default=5_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text())
    n, hidden = meta["n"], meta["hidden_dim"]
    emb_path = args.data_dir / "embeddings.bin"
    print(f"loading {emb_path} as bf16 [{n:,}, {hidden}]")
    emb = load_embeddings(emb_path, n, hidden)

    t0 = time.time()
    print(f"norm stats on {args.norm_sample:,} rows...")
    nstats = norm_stats(emb, args.norm_sample, args.device)
    print(f"  median norm = {nstats['p50']:.3f}  (p05={nstats['p05']:.3f}, p95={nstats['p95']:.3f})")

    print(f"random-pair cosine on {args.pair_sample:,} pairs...")
    pstats = random_pair_cosine(emb, args.pair_sample, args.device)
    print(f"  median pair cosine = {pstats['cos_p50']:.4f}  (p05={pstats['cos_p05']:.4f}, p95={pstats['cos_p95']:.4f})")

    print(f"1-NN cosine distance on {args.knn_sample:,} sampled queries against full cloud...")
    knn = one_nn_distance(emb, args.knn_sample, args.device)
    print(f"  median 1-NN dist = {knn['cos_dist_p50']:.4f}  (p05={knn['cos_dist_p05']:.4f}, p95={knn['cos_dist_p95']:.4f})")

    out = {
        "meta": {"n": n, "hidden_dim": hidden, "elapsed_s": time.time() - t0},
        "norms": nstats,
        "pair_cosine": pstats,
        "one_nn_cos_dist": knn,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  ({out['meta']['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
