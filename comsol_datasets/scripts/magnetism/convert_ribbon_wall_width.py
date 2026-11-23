import sys, numpy as np, csv
from parse_ribbon import load
from scipy.optimize import curve_fit
MU0, Ms = 4e-7 * np.pi, 1.71e6
D = load(sys.argv[1] if len(sys.argv) > 1 else 's5.out')
OUT = sys.argv[2] if len(sys.argv) > 2 else '../../data/magnetism'; x = np.arange(0, 301, 2.0)
out = {'x_nm': x.astype(np.float32)}; rows = []
for cid, c in D['cases'].items():
    pr = np.array(D[(cid, 'PR')], dtype=np.float32); n = (pr.shape[1] - 1) // 3
    prof = np.stack([pr[:, 1:1 + n], pr[:, 1 + n:1 + 2 * n], pr[:, 1 + 2 * n:]], -1)
    out[cid] = prof; out[cid + '_t_ns'] = pr[:, 0]
    HK, Ap = float(c[3]), float(c[-3])
    f = lambda xx, x0, d: -np.tanh((xx - x0) / d)
    (x0, d), _ = curve_fit(f, x, prof[-1, :, 0], p0=[150, 20])
    KJ, AJ = 0.5 * MU0 * Ms * HK, 0.5 * MU0 * Ms * Ap
    mx = np.array(D[(cid, 'MX')])[:, 1].max()
    rows.append([cid, HK, KJ, AJ, np.sqrt(AJ / KJ) * 1e9, d, x0, mx, D['time'][cid]])
    print(cid, f'HK={HK:g} A_J={AJ:.3g} K_J={KJ:.3g} Delta theory {np.sqrt(AJ/KJ)*1e9:.2f} fit {d:.2f} nm  x0 {x0:.1f}')
np.savez_compressed(OUT + '/ribbon_wall_width_comsol.npz', **out)
with open(OUT + '/ribbon_wall_width_comsol.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['case_id', 'HK_A_per_m', 'K_J_per_m3', 'A_J_per_m', 'Delta_theory_nm', 'Delta_fit_nm', 'x0_fit_nm', 'max_norm_error', 'solve_time_s'])
    for r in rows: w.writerow([r[0]] + [f'{v:.6g}' for v in r[1:]])
