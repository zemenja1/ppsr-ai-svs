import numpy as np, sys
def load(fn):
    D = {}
    for l in open(fn):
        k = l.split(",", 2)
        if k[0] in ("AV", "MX", "PR"):
            D.setdefault((k[1], k[0]), []).append([float(v) for v in k[2].split(",")])
        elif k[0] == "CASE":
            D.setdefault("cases", {})[k[1]] = l.strip().split(",")[1:]
        elif k[0] == "SOLVETIME":
            D.setdefault("time", {})[k[1]] = float(l.split(",")[2])
    return D
