"""DEVSIM (with self-heating) Gummel sweeps of dataset samples, or of the library device.

    python devsim_gummel.py b0001 b0002 ...   -> devsim_<sample_id>.csv (columns as bjt_thermal_devsim.csv)
    python devsim_gummel.py library           -> devsim_library.csv   (T0 = 293.15 K, R = 100 K/W, V_C = 3 V)

Needs devsim, numpy, scipy, pandas. About 40 min per sample (one CPU core); run several in parallel.
"""
import os, sys
import pandas as pd
from devsim_bjt_thermal import BJTThermal

DATA = os.environ.get("PPSR_BJT_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "bjt_thermal"))
COLS = ["sample_id", "T0_K", "R_K_W", "VC_V", "NB_cm3", "Nepi_cm3"]
g = pd.read_csv(os.path.join(DATA, "bjt_thermal_gummel.csv"))
for sid in sys.argv[1:]:
    if sid == "library":
        p = pd.Series(dict(zip(COLS, ["library", 293.15, 100.0, 3.0, 9e17, 2e16])))
        vbs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    else:
        c = g[(g.sample_id == sid) & (g.VB_V >= 0.5)]
        p = c.iloc[0]
        vbs = sorted(c.VB_V.unique())
    b = BJTThermal(p.T0_K, p.NB_cm3, p.Nepi_cm3, R=p.R_K_W)
    b.ramp("collector", p.VC_V, 0.25)
    rows = []
    for sweep, thermal in (("isothermal", False), ("selfheating", True)):
        b.ramp("base", 0.5, 0.1)  # the self-heating sweep restarts from 0.5 V (T is still T0 there)
        for vb in vbs:
            b.ramp("base", vb, 0.05, thermal=thermal)
            i = b.currents()
            rows.append({**{k: p[k] for k in COLS}, "sweep": sweep, "VB_V": vb, "IB_A": i["base"],
                         "IC_A": i["collector"], "Tmax_K": b.temperature().max()})
    b.close()
    pd.DataFrame(rows).to_csv(f"devsim_{sid}.csv", index=False, float_format="%.7g")
    print(sid, "done", flush=True)
