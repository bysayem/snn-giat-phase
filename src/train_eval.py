"""Chunking, training, streaming evaluation, detection delay and metrics."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from settings import WARM, SKIP
from gait_data import FS, PHASES


# ------------------------------------------------------------------ data
def chunk(x, y, L=100, stride=50):
    idx = range(0, len(x) - L + 1, stride)
    return (np.stack([x[i:i + L] for i in idx]), np.stack([y[i:i + L] for i in idx]))


def make_chunks(inputs, labels):
    X, Y = zip(*[chunk(x, y) for x, y in zip(inputs, labels)])
    X, Y = np.concatenate(X), np.concatenate(Y)
    Xt = torch.from_numpy(X).float().permute(1, 0, 2).contiguous()   # [T,N,F]
    return Xt, torch.from_numpy(Y).long().T.contiguous()             # [T,N]


# -------------------------------------------------------------- training
def train(model, Xtr, Ytr, Xva, Yva, cfg, lr, rate_reg, name, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    N = Xtr.shape[1]
    best, best_state = -1, None
    hist = {"loss": [], "val_acc": []}
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, cfg.batch):
            b = perm[i:i + cfg.batch]
            logits, _, st = model(Xtr[:, b])
            y = Ytr[:, b]
            loss = F.cross_entropy(logits[WARM:].reshape(-1, 4), y[WARM:].reshape(-1))
            if st and rate_reg > 0:
                loss = loss + rate_reg * (st["rate1"] + st["rate2"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            lv, _, _ = model(Xva)
            acc = (lv[WARM:].argmax(-1) == Yva[WARM:]).float().mean().item()
        hist["loss"].append(tot / N)
        hist["val_acc"].append(acc)
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == cfg.epochs):
            print(f"[{name}] ep {ep+1:02d}/{cfg.epochs} loss {tot/N:.4f} val_acc {acc:.4f}")
    model.load_state_dict(best_state)
    return best, hist


# ------------------------------------------------------------ evaluation
@torch.no_grad()
def stream_eval(model, inputs):
    """Each subject = ONE continuous stream with persistent state."""
    model.eval()
    preds, r1, r2 = [], [], []
    for x in inputs:
        lg, _, st = model(torch.from_numpy(x).unsqueeze(1))
        preds.append(lg.squeeze(1).argmax(-1).numpy())
        if st:
            r1.append(st["rate1"].item())
            r2.append(st["rate2"].item())
    return preds, (float(np.mean(r1)) if r1 else None), (float(np.mean(r2)) if r2 else None)


def detection_delay(yt, yp, max_lag=30):
    """Samples between a true transition and the first prediction of the new phase."""
    delays, missed = [], 0
    for i in np.where(np.diff(yt) != 0)[0] + 1:
        c, found = yt[i], False
        for j in range(i, min(i + max_lag, len(yt))):
            if yt[j] != c:
                break
            if yp[j] == c:
                delays.append(j - i)
                found = True
                break
        if not found:
            missed += 1
    return delays, missed


def metrics(ys, ps):
    yt = np.concatenate([y[SKIP:] for y in ys])
    yp = np.concatenate([p[SKIP:] for p in ps])
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (yt, yp), 1)
    f1 = []
    for c in range(4):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    all_d, all_m = [], 0
    for y, p in zip(ys, ps):
        d, m = detection_delay(y[SKIP:], p[SKIP:])
        all_d += d
        all_m += m
    n_tr = len(all_d) + all_m
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": {PHASES[i]: float(f1[i]) for i in range(4)},
        "mean_detection_delay_ms": float(np.mean(all_d) * 1000 / FS) if all_d else None,
        "p95_detection_delay_ms": float(np.percentile(all_d, 95) * 1000 / FS) if all_d else None,
        "missed_transitions_pct": float(100 * all_m / max(n_tr, 1)),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def time_per_step(model, x, n=1000):
    """CPU wall-clock per time step (microseconds). Informational only."""
    model.eval()
    xs = torch.from_numpy(x[:n + 50]).unsqueeze(1)
    st = model.init_state(1) if hasattr(model, "init_state") else None
    for t in range(50):
        _, st, _ = model(xs[t:t + 1], st)
    t0 = time.perf_counter()
    for t in range(50, n + 50):
        _, st, _ = model(xs[t:t + 1], st)
    return (time.perf_counter() - t0) / n * 1e6


def fmt(v, spec=".1f"):
    return "n/a" if v is None else format(v, spec)
