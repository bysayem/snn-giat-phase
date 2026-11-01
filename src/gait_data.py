"""Synthetic gait generator (subject-wise variability) for gait phase detection."""
import numpy as np

FS = 100  # Hz
PHASES = ["Loading", "MidStance", "PreSwing", "Swing"]
BOUNDS = np.array([0.15, 0.50, 0.62])   # gait-cycle fraction where phase changes
CHANNELS = ["hip", "knee", "ankle", "shank_gyro"]


def phase_of(p):
    """p in [0,1) -> class id 0..3"""
    return np.digitize(p, BOUNDS)


def _g(p, mu, sig, amp):
    return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)


def joint_angles(p):
    hip = 10 + 20 * np.cos(2 * np.pi * p)
    knee = 5 + _g(p, 0.15, 0.05, 12) + _g(p, 0.72, 0.09, 55)
    ankle = (_g(p, 0.02, 0.03, -5) + _g(p, 0.40, 0.12, 10)
             + _g(p, 0.60, 0.05, -18) + _g(p, 0.85, 0.10, 4))
    return np.stack([hip, knee, ankle], axis=1)


def make_subject(seed, n_strides=60):
    """Returns x [N,4] (hip, knee, ankle [deg], shank gyro [deg/s]) and y [N]."""
    rng = np.random.default_rng(seed)
    base_T = rng.uniform(0.95, 1.30)          # stride duration (s)
    gain = rng.uniform(0.85, 1.15, 3)         # subject-specific ROM
    offs = rng.normal(0, 2.0, 3)              # sensor offset (deg)
    ang_noise = rng.uniform(0.7, 1.8)
    gyr_noise = rng.uniform(8, 20)

    angs, labs = [], []
    for _ in range(n_strides):
        T = base_T * rng.uniform(0.92, 1.08)  # stride-to-stride variability
        n = int(round(T * FS))
        p = np.arange(n) / n
        a = joint_angles(p) * gain * rng.uniform(0.95, 1.05, 3) + offs
        angs.append(a)
        labs.append(phase_of(p))
    ang = np.concatenate(angs)
    y = np.concatenate(labs)

    shank = ang[:, 0] - ang[:, 1]
    gyro = np.gradient(shank) * FS
    x = np.column_stack([ang, gyro])
    x[:, :3] += rng.normal(0, ang_noise, (len(x), 3))
    x[:, 3] += rng.normal(0, gyr_noise, len(x))
    return x.astype(np.float32), y.astype(np.int64)
