"""run(): train + evaluate both models. Plus summary, plots and saving to disk."""
import os, json, random
from dataclasses import asdict
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import Cfg
from gait_data import FS, PHASES, CHANNELS
from encoding import build_data
from models import GaitSNN, GaitLSTM
from energy import lstm_cost, snn_cost
from train_eval import make_chunks, train, stream_eval, metrics, time_per_step, fmt


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run(cfg, verbose=True):
    set_seed(cfg.seed)
    D, n_in = build_data(cfg)
    Xs_tr, Y_tr = make_chunks(D["train"]["spk"], D["train"]["y"])
    Xs_va, Y_va = make_chunks(D["val"]["spk"], D["val"]["y"])
    Xr_tr, _ = make_chunks(D["train"]["raw"], D["train"]["y"])
    Xr_va, _ = make_chunks(D["val"]["raw"], D["val"]["y"])

    snn, lstm = GaitSNN(n_in, cfg.hidden, 4), GaitLSTM(4, cfg.hidden, 4)
    va_snn, h_snn = train(snn, Xs_tr, Y_tr, Xs_va, Y_va, cfg, cfg.lr_snn, cfg.rate_reg, "SNN", verbose)
    va_lstm, h_lstm = train(lstm, Xr_tr, Y_tr, Xr_va, Y_va, cfg, cfg.lr_lstm, 0.0, "LSTM", verbose)

    ys = D["test"]["y"]
    p_snn, r1, r2 = stream_eval(snn, D["test"]["spk"])
    p_lstm, _, _ = stream_eval(lstm, D["test"]["raw"])
    in_spk = float(np.mean([s.sum(1).mean() for s in D["test"]["spk"]]))

    res = {"config": asdict(cfg),
           "snn": {"params": sum(p.numel() for p in snn.parameters()), "val_acc": va_snn,
                   "metrics": metrics(ys, p_snn), "avg_input_spikes_per_step": in_spk,
                   "n_input_neurons": n_in, "hidden_rate1": r1, "hidden_rate2": r2,
                   "energy": snn_cost(cfg.hidden, 4, in_spk, r1, r2),
                   "cpu_us_per_step": time_per_step(snn, D["test"]["spk"][0])},
           "lstm": {"params": sum(p.numel() for p in lstm.parameters()), "val_acc": va_lstm,
                    "metrics": metrics(ys, p_lstm),
                    "energy": lstm_cost(4, cfg.hidden, 4),
                    "cpu_us_per_step": time_per_step(lstm, D["test"]["raw"][0])}}
    res["energy_ratio_lstm_over_snn"] = (res["lstm"]["energy"]["energy_pj"]
                                         / res["snn"]["energy"]["energy_pj"])
    art = {"D": D, "snn": snn, "lstm": lstm, "p_snn": p_snn, "p_lstm": p_lstm,
           "hist": {"snn": h_snn, "lstm": h_lstm}}
    return res, art


def print_summary(res):
    print("\n=========== RESULTS (test, unseen subjects) ===========")
    for k in ["snn", "lstm"]:
        m = res[k]["metrics"]
        print(f"{k.upper():5s} acc {m['accuracy']:.3f} | macroF1 {m['macro_f1']:.3f} | "
              f"delay {fmt(m['mean_detection_delay_ms'])} ms (p95 {fmt(m['p95_detection_delay_ms'])}) | "
              f"missed {m['missed_transitions_pct']:.1f}% | params {res[k]['params']:,} | "
              f"energy {res[k]['energy']['energy_pj']:,.0f} pJ/step | "
              f"CPU {res[k]['cpu_us_per_step']:.0f} us/step")
    s = res["snn"]
    print(f"SNN firing rate: L1 {s['hidden_rate1']:.3f}, L2 {s['hidden_rate2']:.3f} spikes/neuron/step; "
          f"input {s['avg_input_spikes_per_step']:.2f}/{s['n_input_neurons']} spikes/step")
    print(f"Estimated energy ratio LSTM/SNN: {res['energy_ratio_lstm_over_snn']:.1f}x")


def make_plots(res, art, out_dir, a=300, b=600, show=True):
    os.makedirs(out_dir, exist_ok=True)
    D = art["D"]
    x_raw, spk, y = D["test"]["raw"][0], D["test"]["spk"][0], D["test"]["y"][0]
    t = np.arange(a, b) / FS

    # Figure 1: timeline
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for c, n in enumerate(CHANNELS):
        ax[0].plot(t, x_raw[a:b, c], label=n, lw=1)
    ax[0].legend(ncol=4, fontsize=8); ax[0].set_ylabel("normalized signal")
    ts, ns = np.nonzero(spk[a:b])
    ax[1].scatter(t[ts], ns, s=6, marker="|", color="k")
    ax[1].set_ylabel("input neuron id")
    ax[1].set_title(f"Input spike raster ({spk[a:b].sum() / len(t):.2f} spikes/step)", fontsize=9)
    for k, (p, nm) in enumerate([(art["p_snn"][0], "SNN"), (art["p_lstm"][0], "LSTM")]):
        ax[2 + k].step(t, y[a:b], where="post", label="truth", lw=2, alpha=0.6)
        ax[2 + k].step(t, p[a:b], where="post", label=nm, ls="--")
        ax[2 + k].set_yticks(range(4)); ax[2 + k].set_yticklabels(PHASES, fontsize=7)
        ax[2 + k].legend(loc="upper right", fontsize=8)
    ax[3].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_timeline.png", dpi=150)
    plt.show() if show else plt.close()

    # Figure 2: summary
    fig, ax = plt.subplots(2, 2, figsize=(12, 8)); ax = ax.ravel()
    e = [res["lstm"]["energy"]["energy_pj"], res["snn"]["energy"]["energy_pj"]]
    ax[0].bar(["LSTM", "SNN"], e, color=["tab:red", "tab:green"]); ax[0].set_yscale("log")
    ax[0].set_ylabel("pJ / time step (log)"); ax[0].set_title("Estimated energy (Horowitz 2014)")
    for i, v in enumerate(e):
        ax[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    d = [res["lstm"]["metrics"]["mean_detection_delay_ms"] or 0,
         res["snn"]["metrics"]["mean_detection_delay_ms"] or 0]
    ax[1].bar(["LSTM", "SNN"], d, color=["tab:red", "tab:green"])
    ax[1].set_ylabel("ms"); ax[1].set_title("Mean transition detection delay")
    for i, v in enumerate(d):
        ax[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    cm = np.array(res["snn"]["metrics"]["confusion_matrix"], float)
    cm /= cm.sum(1, keepdims=True)
    ax[2].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax[2].set_xticks(range(4)); ax[2].set_yticks(range(4))
    ax[2].set_xticklabels(PHASES, rotation=30, fontsize=7); ax[2].set_yticklabels(PHASES, fontsize=7)
    for i in range(4):
        for j in range(4):
            ax[2].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[2].set_title("SNN confusion (row-normalized)")
    ax[3].plot(art["hist"]["snn"]["val_acc"], label="SNN")
    ax[3].plot(art["hist"]["lstm"]["val_acc"], label="LSTM")
    ax[3].set_xlabel("epoch"); ax[3].set_ylabel("val accuracy"); ax[3].legend()
    ax[3].set_title("Validation accuracy")
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_summary.png", dpi=150)
    plt.show() if show else plt.close()


def save_run(res, art, out_dir, plots=True, show=True):
    """Saves results.json, model checkpoints and figures into out_dir (e.g. on Drive)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(res, f, indent=2)
    torch.save(art["snn"].state_dict(), f"{out_dir}/snn.pt")
    torch.save(art["lstm"].state_dict(), f"{out_dir}/lstm.pt")
    if plots:
        make_plots(res, art, out_dir, show=show)
    print("saved ->", out_dir)


def to_row(name, r):
    m, s = r["snn"]["metrics"], r["snn"]
    return {"setting": name,
            "SNN acc": m["accuracy"], "SNN F1": m["macro_f1"],
            "SNN delay ms": m["mean_detection_delay_ms"],
            "LSTM acc": r["lstm"]["metrics"]["accuracy"],
            "LSTM delay ms": r["lstm"]["metrics"]["mean_detection_delay_ms"],
            "in spikes/step": s["avg_input_spikes_per_step"],
            "rate L1": s["hidden_rate1"], "rate L2": s["hidden_rate2"],
            "SNN pJ": s["energy"]["energy_pj"], "LSTM pJ": r["lstm"]["energy"]["energy_pj"],
            "ratio": r["energy_ratio_lstm_over_snn"]}
