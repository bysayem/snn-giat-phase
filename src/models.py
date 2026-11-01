"""Recurrent LIF spiking network (surrogate gradient) and LSTM baseline."""
import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate gradient backward."""
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (ctx.slope * x.abs() + 1.0) ** 2, None


class LIF(nn.Module):
    """Leaky integrate-and-fire, reset-by-subtraction, learnable per-neuron decay."""
    def __init__(self, n, thr=1.0, slope=10.0):
        super().__init__()
        self.b_logit = nn.Parameter(torch.logit(torch.linspace(0.8, 0.99, n)))
        self.thr, self.slope = thr, slope

    def forward(self, cur, mem):
        beta = torch.sigmoid(self.b_logit)
        mem = beta * mem + cur
        spk = SpikeFn.apply(mem - self.thr, self.slope)
        mem = mem - spk.detach() * self.thr
        return spk, mem


class GaitSNN(nn.Module):
    def __init__(self, n_in=48, n_h=64, n_out=4):
        super().__init__()
        self.n_in, self.n_h, self.n_out = n_in, n_h, n_out
        self.fc1 = nn.Linear(n_in, n_h)
        self.rec1 = nn.Linear(n_h, n_h, bias=False)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_out)
        self.lif1, self.lif2 = LIF(n_h), LIF(n_h)
        self.out_logit = nn.Parameter(torch.tensor(0.8).logit())
        nn.init.normal_(self.fc1.weight, 0, 0.5)
        nn.init.normal_(self.rec1.weight, 0, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.3)

    def init_state(self, B, device="cpu"):
        z = lambda n: torch.zeros(B, n, device=device)
        return (z(self.n_h), z(self.n_h), z(self.n_out), z(self.n_h))

    def forward(self, x, state=None):
        """x: [T,B,n_in] spikes -> logits [T,B,n_out], state, stats."""
        T, B = x.shape[:2]
        if state is None:
            state = self.init_state(B, x.device)
        m1, m2, mo, s1 = state
        beta_o = torch.sigmoid(self.out_logit)
        outs, c1, c2 = [], 0.0, 0.0
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(x[t]) + self.rec1(s1), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            mo = beta_o * mo + self.fc3(s2)       # non-spiking leaky readout
            outs.append(mo)
            c1 = c1 + s1.sum()
            c2 = c2 + s2.sum()
        stats = {"rate1": c1 / (T * B * self.n_h),   # spikes / neuron / step
                 "rate2": c2 / (T * B * self.n_h)}
        return torch.stack(outs), (m1, m2, mo, s1), stats


class GaitLSTM(nn.Module):
    def __init__(self, n_in=4, n_h=64, n_out=4):
        super().__init__()
        self.n_in, self.n_h, self.n_out = n_in, n_h, n_out
        self.lstm = nn.LSTM(n_in, n_h)
        self.fc = nn.Linear(n_h, n_out)

    def forward(self, x, state=None):
        y, state = self.lstm(x, state)
        return self.fc(y), state, {}
