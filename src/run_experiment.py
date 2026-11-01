"""CLI:  python run_experiment.py --epochs 40 --out ../results/main"""
import argparse, torch
from settings import Cfg
from experiment import run, print_summary, save_run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate_reg", type=float, default=0.0)
    ap.add_argument("--n_levels", type=int, default=6)
    ap.add_argument("--out", default="../results/main")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = Cfg(epochs=a.epochs, hidden=a.hidden, seed=a.seed,
              rate_reg=a.rate_reg, n_levels=a.n_levels)
    res, art = run(cfg)
    print_summary(res)
    save_run(res, art, a.out, show=False)
