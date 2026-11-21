"""Convert the COMSOL batch log of the generator yoke sweep into dataset files."""
import sys, numpy as np, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else 'generator.log'
OUT = sys.argv[2] if len(sys.argv) > 2 else '../../data/generator_2d'
F, E, L = {}, {}, []
for line in open(src, encoding='utf-8', errors='replace'):
    p = line.strip().split(',')
    if p[0] == 'FIELD': F.setdefault(float(p[1]), {})['solve_time_s'] = float(p[2])
    elif p[0] == 'EMF':
        v = np.array(p[2:], float); n = len(v) // 2; E[float(p[1])] = (v[n:], v[:n])  # COMSOL returns all EMF values, then all times
    elif p[0] in ('BMAXS', 'BAVS', 'MURS', 'BMAXR'):
        F.setdefault(float(p[1]), {})[p[0]] = np.array(p[2:], float)
    elif p[0] == 'LOSS':
        L.append([float(x) for x in p[1:8]])
rows = []
for k, d in sorted(F.items()):
    t, e = E[k]
    rows.append(dict(kmu=k, mur_stator_mean=d['MURS'].mean(), B_stator_max_T=d['BMAXS'].max(),
                     B_stator_mean_T=d['BAVS'].mean(), B_rotor_max_T=d['BMAXR'].max(),
                     emf_peak_V=np.abs(e).max(), emf_rms_V=np.sqrt(np.mean(e[1:] ** 2)),
                     solve_time_s=d['solve_time_s']))
field = pd.DataFrame(rows)
L = pd.DataFrame(L, columns=['kmu', 'sigma_S_per_m', 'd_mm', 'P_stator_W', 'P_rotor_W', 'P_shaft_W', 'P_magnets_W'])
L['P_iron_W'] = L.P_stator_W + L.P_rotor_W + L.P_shaft_W
hyst = L[L.sigma_S_per_m == 0].set_index('kmu').P_iron_W
thin = L[L.d_mm < 1e-3].set_index(['kmu', 'sigma_S_per_m']).P_iron_W
main = L[L.d_mm >= 1e-3].copy()
main['P_hyst_W'] = main.kmu.map(hyst)
th = np.array([thin.get((k, s), np.nan) if s > 0 else hyst[k] for k, s in zip(main.kmu, main.sigma_S_per_m)])
main['P_excess_W'] = th - main.P_hyst_W
main['P_classical_W'] = main.P_iron_W - th
main = main.merge(field[['kmu', 'mur_stator_mean', 'emf_peak_V']], on='kmu')
main['P_total_W'] = main.P_iron_W + main.P_magnets_W
cols = ['kmu', 'mur_stator_mean', 'sigma_S_per_m', 'd_mm', 'emf_peak_V', 'P_stator_W', 'P_rotor_W', 'P_shaft_W',
        'P_iron_W', 'P_hyst_W', 'P_classical_W', 'P_excess_W', 'P_magnets_W', 'P_total_W']
main = main[cols].sort_values(['kmu', 'sigma_S_per_m', 'd_mm'])
emf = pd.concat([pd.DataFrame(dict(kmu=k, t_s=t, emf_V=e)) for k, (t, e) in sorted(E.items())])
field.to_csv(OUT + '/generator_field_cases.csv', index=False, float_format='%.6g')
main.to_csv(OUT + '/generator_losses.csv', index=False, float_format='%.6g')
emf.to_csv(OUT + '/generator_emf.csv', index=False, float_format='%.6g')
print(field.round(3).to_string(index=False)); print(len(main), 'loss rows')
