"""Spin-transfer-torque switching of a perpendicular MTJ free layer (40 nm CoFeB-like disc), magnum.np.

Writes stt_disc_J<f>.npz (f = J/Jc0) with the average magnetization <m>(t) every 10 ps and m_z snapshots
every 50 ps. Used by notebooks/04_magnetism (section on STT-MRAM). Each run takes ~5-10 min on a CPU.

    python stt_disc_magnumnp.py <output_dir> [f1 f2 ...]
"""
import os, sys, time
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np
import torch
from magnumnp import (Mesh, State, DemagField, ExchangeField, UniaxialAnisotropyField, SpinTorqueSlonczewski,
                      LLGSolver, normalize)

MU0, HBAR, E = 4e-7 * np.pi, 1.054571817e-34, 1.602176634e-19
# free layer: Ms, thickness, damping, effective anisotropy (incl. thin-film demag), polarization, Lambda
Ms, t, alpha, Keff, P, Lam, A = 1.0e6, 1.3e-9, 0.01, 2.0e5, 0.6, 1.0, 1.5e-11
HK = 2 * Keff / (MU0 * Ms)
eps = P * Lam**2 / (Lam**2 + 1)
Jc0 = alpha * HK * E * Ms * t * MU0 / (HBAR * eps)       # macrospin critical current density (A/m^2)
T_END, DT_OUT, DT_SNAP = 15e-9, 1e-11, 5e-11

out = sys.argv[1] if len(sys.argv) > 1 else "."
factors = [float(a) for a in sys.argv[2:]] or [1.5, 2.0, 3.0, 4.0]

torch.set_default_dtype(torch.float64)
n, dx = (16, 16, 1), (2.5e-9, 2.5e-9, t)
mesh = Mesh(n, dx)
state = State(mesh)
x, y, z = state.SpatialCoordinate()
disc = (x - 20e-9)**2 + (y - 20e-9)**2 < 20e-9**2
Ku = Keff + 0.5 * MU0 * Ms**2                # bulk anisotropy; the demag field of the thin disc subtracts ~mu0 Ms^2/2
Msf, Af, Kuf = state.Constant(0.0), state.Constant(0.0), state.Constant(0.0)   # material only inside the disc
Msf[disc], Af[disc], Kuf[disc] = Ms, A, Ku          # (A = 0 outside: no exchange coupling to empty cells)
state.material = {"Ms": Msf, "A": Af, "Ku": Kuf, "Ku_axis": [0, 0, 1], "alpha": alpha, "P": P, "Lambda": Lam,
                  "epsilon_prime": 0.0, "mp": state.Constant([0, 0, 1.0]), "d": t, "J": 0.0}
terms = [DemagField(), ExchangeField(), UniaxialAnisotropyField(), SpinTorqueSlonczewski()]
mask = disc[..., 0].cpu().numpy()
nd = disc.sum().item()
print(f"Jc0 (macrospin) = {Jc0 / 1e10:.2f} MA/cm^2")
for f in factors:
    state.m = state.Constant([0.05, 0, -1.0]); normalize(state.m); state.t = 0.0   # free layer antiparallel to p
    state.material["J"] = f * Jc0
    llg = LLGSolver(terms)
    t0 = time.time(); ts, ms, snaps = [0.0], [[0.05, 0.0, -1.0]], [state.m[:, :, 0, 2].cpu().numpy()]
    k = 0
    while state.t < T_END - 1e-15:
        llg.step(state, DT_OUT); k += 1
        ts.append(state.t if isinstance(state.t, float) else float(state.t))
        ms.append([(state.m[..., i] * disc).sum().item() / nd for i in range(3)])
        if k % round(DT_SNAP / DT_OUT) == 0:
            snaps.append(state.m[:, :, 0, 2].cpu().numpy())
    ms = np.array(ms)
    fn = os.path.join(out, f"stt_disc_J{f:.1f}.npz")
    np.savez_compressed(fn, t=np.array(ts), m=ms.astype(np.float32), mz_snap=np.array(snaps, dtype=np.float16),
                        mask=mask, J=f * Jc0, Jc0=Jc0, dx=2.5e-9, dt_snap=DT_SNAP)
    sw = ts[np.argmax(ms[:, 2] > 0)] if (ms[:, 2] > 0).any() else np.nan
    print(f"J = {f} Jc0: first <mz> > 0 at {sw * 1e9:.2f} ns, final <mz> = {ms[-1, 2]:.2f} ({time.time() - t0:.0f} s)",
          flush=True)
