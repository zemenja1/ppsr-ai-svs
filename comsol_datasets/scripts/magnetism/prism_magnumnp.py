"""Step 4 reference data: magnum.np loops of 1:1:4 Fe prisms of growing size, and the energy of uniform,
two-domain and vortex states versus size. Usage: python prism_magnumnp.py loops | states <widths in nm, e.g. 10,20,40> <tag>   (output folder: $OUT, default mm4)"""
import os, sys, time, numpy as np, torch
from magnumnp import Mesh, State, DemagField, ExchangeField, ExternalField, MinimizerBB, set_log_level
set_log_level(40); torch.set_num_threads(5)
Ms, A = 1.71e6, 2.1e-11
OUT = os.environ.get('OUT', 'mm4')

def prism(w, cell):
    n = int(round(w / cell))
    state = State(Mesh((n, n, 4 * n), (w / n, w / n, w / n)))
    state.material = {"Ms": Ms, "A": A}
    return state

def loop(w, cell, dH=10e3, Hmax=700e3, tilt=0.5):
    state = prism(w, cell); state.m = state.Constant([0, 0, 1])
    demag, exch = DemagField(), ExchangeField()
    ext = ExternalField(state.Constant([0, 0, 0]))
    mini = MinimizerBB([demag, exch, ext])
    th = np.deg2rad(tilt); rows = []; snap = None; prev = None
    for H in np.arange(0, -Hmax - 1, -dH):
        ext.h = state.Constant([H * np.sin(th), 0, H * np.cos(th)])
        mini.minimize(state, maxiter=50000, dm_tol=1.0)
        mm = state.m.mean(dim=(0, 1, 2)).tolist()
        rows.append([H] + mm + [float(demag.E(state)), float(exch.E(state))])
        if mm[2] < 0 and snap is None:
            snap = prev                      # last state before the switch
        prev = state.m.detach().cpu().numpy().copy()
        if mm[2] < -0.95:
            break
    return np.array(rows), snap

def relax_state(w, cell, kind):
    state = prism(w, cell)
    x, y, z = [(np.arange(s) + 0.5) * d - s * d / 2 for s, d in zip(state.mesh.n, state.mesh.dx)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    if kind == 'uniform':
        m = np.stack([0 * X, 0 * X, 1 + 0 * X], -1)
    elif kind == 'two_domain':
        m = np.stack([0 * X, 0.1 * np.exp(-(X / (w / 10))**2), np.sign(X + 1e-15)], -1)
    else:  # vortex around the long axis
        r = np.hypot(X, Y); rc = w / 8
        m = np.stack([-Y, X, rc * np.exp(-(r / rc)**2) * 4 + 1e-3 * w], -1)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    state.m = state.Constant([0, 0, 1]); state.m[...] = torch.tensor(m, dtype=state.m.dtype)
    demag, exch = DemagField(), ExchangeField()
    MinimizerBB([demag, exch]).minimize(state, maxiter=100000, dm_tol=0.1)
    V = w * w * 4 * w
    Ed, Ex = float(demag.E(state)), float(exch.E(state))
    return Ed / V, Ex / V, state.m.mean(dim=(0, 1, 2)).tolist(), state.m.detach().cpu().numpy()

if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'loops':
        for w, cell in [(5e-9, 1e-9), (10e-9, 1.25e-9), (20e-9, 2e-9), (40e-9, 2.5e-9), (80e-9, 3.33e-9)]:
            t = time.time(); rows, snap = loop(w, cell)
            np.savetxt(f'{OUT}/loop_w{w*1e9:.0f}nm.csv', rows, delimiter=',', comments='',
                       header='H_A_per_m,mx,my,mz,E_demag_J,E_exch_J', fmt='%.6g')
            np.savez_compressed(f'{OUT}/snap_w{w*1e9:.0f}nm.npz', m=snap.astype(np.float32))
            print('LOOP', w, cell, len(rows), rows[-1][0], time.time() - t, flush=True)
    else:
        res = []
        for w in [float(v) * 1e-9 for v in sys.argv[2].split(',')]:
            cell = min(2.5e-9, w / 8) if w <= 40e-9 else 4e-9
            for kind in ['uniform', 'two_domain', 'vortex']:
                t = time.time(); ed, ex, mm, m = relax_state(w, cell, kind)
                res.append([w, kind, ed, ex, ed + ex] + mm)
                np.savez_compressed(f'{OUT}/state_{kind}_w{w*1e9:.0f}nm.npz', m=m.astype(np.float32))
                print('STATE', w, kind, ed, ex, ed + ex, mm, time.time() - t, flush=True)
                with open(f'{OUT}/states_{sys.argv[3]}.csv', 'a') as f:
                    f.write(f'{w*1e9:.4g},{kind},{ed:.6g},{ex:.6g},{ed+ex:.6g},{mm[0]:.4g},{mm[1]:.4g},{mm[2]:.4g}\n')
