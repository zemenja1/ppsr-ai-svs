"""Parse a transformer batch log -> transformer_cases.csv (cycle averages over the last period) and
transformer_waveforms.npz (one period of u_1, i_1, u_2, i_2 and the volume-averaged |B| of the core).
Usage: python convert_transformer.py <out_dir> <log> [<log> ...]   (several logs are merged; first result wins)"""
import sys, csv, numpy as np


def load(fn):
    cases, D, times = {}, {}, {}
    for line in open(fn, errors="replace"):
        p = line.strip().split(",")
        if p[0] == "CASE":
            k = p[1].split(";"); cases[k[0]] = dict(Vac=float(k[1]), nu=float(k[2]), Rs=float(k[3]))
        elif p[0] == "SOLVETIME":
            times[p[1]] = float(p[2]) + float(p[3])
        elif p[0] in ("GV", "MX", "AV", "LQ", "RC"):
            D.setdefault((p[1], p[0]), []).append([float(v) for v in p[2:]])
    return cases, D, times


def summarize(cid, c, D, times, Rp=100.0):
    gv = np.array(D[(cid, "GV")]); t = gv[:, 0]; T = 1 / c["nu"]
    last = t >= 1.5 * T - 1e-9 * T                       # last period: 1.5 T ... 2.5 T (41 points incl. both ends)
    g = gv[last][:-1]                                     # drop the duplicate end point
    V1, I1, V2, I2 = g[:, 1], g[:, 2], g[:, 3], g[:, 4]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    R1, R2 = D[(cid, "RC")][-1][-2:]
    mx = np.array(D[(cid, "MX")]); av = np.array(D[(cid, "AV")])
    P_load = c["Rs"] * np.mean(I2 ** 2)
    P_cu = R1 * np.mean(I1 ** 2) + R2 * np.mean(I2 ** 2)
    P_core = D[(cid, "LQ")][-1][-1]
    return dict(case_id=cid, Vac_V=c["Vac"], f_Hz=c["nu"], Rload_ohm=c["Rs"],
                V1_rms_V=rms(V1), I1_rms_A=rms(I1), V2_rms_V=rms(V2), I2_rms_A=rms(I2),
                P_load_W=P_load, P_core_W=P_core, P_copper_W=P_cu, P_Rp_W=Rp * np.mean(I1 ** 2),
                efficiency=P_load / (P_load + P_core + P_cu),
                B_max_T=float(mx[last, 1].max()), B_mean_T=float(av[last, 1].max()),
                H_mean_peak_A_per_m=float(av[last, 2].max()), solve_time_s=times.get(cid, np.nan)), \
        dict(t=(t[last][:-1] - 1.5 * T) / T, V1=V1, I1=I1, V2=V2, I2=I2, B=av[last][:-1, 1])


if __name__ == "__main__":
    cases, D, times = {}, {}, {}
    for fn in sys.argv[2:]:
        c1, D1, t1 = load(fn)
        done = [k for k in c1 if (k, "LQ") in D1 and k not in times]
        cases.update({k: c1[k] for k in done}); times.update({k: t1[k] for k in done})
        D.update({key: v for key, v in D1.items() if key[0] in done})
    rows, wav = [], {}
    for cid, c in sorted(cases.items()):
        if (cid, "LQ") not in D:
            print("missing", cid); continue
        r, w = summarize(cid, c, D, times); rows.append(r)
        for k, v in w.items():
            wav[f"{cid}_{k}"] = np.asarray(v, dtype=np.float32)
    out = sys.argv[1]
    with open(out + "/transformer_cases.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(out + "/transformer_waveforms.npz", **wav)
    for r in rows:
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
