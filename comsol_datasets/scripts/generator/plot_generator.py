import sys, pandas as pd, numpy as np, matplotlib
OUT = sys.argv[1] if len(sys.argv) > 1 else '../../data/generator_2d'
matplotlib.use('Agg'); import matplotlib.pyplot as plt
f = pd.read_csv(OUT + '/generator_field_cases.csv'); L = pd.read_csv(OUT + '/generator_losses.csv'); E = pd.read_csv(OUT + '/generator_emf.csv')
BL, OR = '#1f4e8c', '#e07b10'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
r = L[(L.kmu == 1) & (L.d_mm == 0.35)].sort_values('sigma_S_per_m'); s = r.sigma_S_per_m / 1e6
ax[0].stackplot(s, r.P_hyst_W, r.P_excess_W, r.P_classical_W, r.P_magnets_W, colors=[BL, OR, '#f2b880', '#9bb7d9'],
                labels=['hysteresis', 'excess', 'classical eddy', 'magnets'])
ax[0].set_xlabel('iron conductivity σ (MS/m)'); ax[0].set_ylabel('loss (W)'); ax[0].legend(loc='upper left', fontsize=8)
ax[0].set_title('Reference B(H) curve, d = 0.35 mm')
for d, c in zip(sorted(L.d_mm.unique()), plt.cm.Oranges(np.linspace(0.4, 1, L.d_mm.nunique()))):
    q = L[(L.kmu == 1) & (L.d_mm == d) & (L.sigma_S_per_m > 0)]
    ax[1].plot(q.sigma_S_per_m / 1e6, q.P_classical_W, 'o-', color=c, label=f'd = {d} mm')
ax[1].set_xlabel('iron conductivity σ (MS/m)'); ax[1].set_ylabel('classical eddy loss (W)'); ax[1].legend(fontsize=8)
ax[1].set_title('Classical eddy loss vs lamination')
q = L[(L.sigma_S_per_m == 1e7) & (L.d_mm == 0.35)].sort_values('mur_stator_mean')
ax[2].semilogx(q.mur_stator_mean, q.P_iron_W, 'o-', color=BL, label='iron loss')
ax[2].semilogx(q.mur_stator_mean, q.P_magnets_W, 's-', color='#9bb7d9', label='magnet loss')
ax[2].set_xlabel('mean relative permeability of stator iron'); ax[2].set_ylabel('loss (W)')
a2 = ax[2].twinx(); a2.semilogx(q.mur_stator_mean, q.emf_peak_V, '^-', color=OR, label='EMF peak'); a2.set_ylabel('EMF peak (V)', color=OR)
ax[2].legend(loc='upper left', fontsize=8); ax[2].set_title('σ = 10 MS/m, d = 0.35 mm')
plt.tight_layout(); plt.savefig(OUT + '/generator_yoke_losses.png', dpi=130)
