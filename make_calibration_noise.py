"""
Held-out noise-only blocks for setting detection thresholds (empirical 99th percentile).
Must NOT be the training noise the models were fit on. Same format as train_noise_raw.pt.

    python make_calibration_noise.py              # 10,000 blocks, seed 12345
"""
import argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--length", type=int, default=1024)
ap.add_argument("--seed", type=int, default=12345)
args = ap.parse_args()

g = torch.Generator().manual_seed(args.seed)
raw = torch.randn(args.n, 2, args.length, generator=g, dtype=torch.float32)       # unnormalized
m = raw.mean(dim=(1, 2), keepdim=True)
s = raw.std(dim=(1, 2), keepdim=True)
norm = (raw - m) / (s + 1e-8)                                                     # per-block joint I/Q

Path("spectrum_data").mkdir(exist_ok=True)
torch.save(raw, "spectrum_data/calib_noise_raw.pt")   # for ED / MME / CAV
torch.save(norm, "spectrum_data/calib_noise.pt")      # for Psi-NN / CAE (take channel 0 = I)
print(f"Saved {args.n} held-out noise blocks: spectrum_data/calib_noise_raw.pt, calib_noise.pt")
