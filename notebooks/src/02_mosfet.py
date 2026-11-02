# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Case study 3: the MOSFET – a neural network as a transistor model
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zemenja1/ppsr-ai-svs/blob/main/notebooks/02_mosfet.ipynb)
#
# *Teaching materials of the project “AI ve výuce – Elektrotechnologie”, CTU in Prague, FEE.*
# *Version 1 (October 2026). License: text CC BY 4.0, code MIT.*
#
# **What you will do**
#
# 1. Recall how a MOSFET works and derive its threshold voltage and current from simple physics.
# 2. **Tier 1 – one transistor:** explore the drain current $I_D(V_G, V_D)$ of one device simulated with
#    COMSOL Multiphysics, compare it with the classical textbook model, and train a small neural network that
#    reproduces it.
# 3. **Tier 2 – many transistor designs:** train a network that also takes the *design* of the transistor
#    (doping, oxide thickness, gate length, junction depth, temperature) as input, and check whether it learned
#    the right physics.
#
# **How to work with this notebook:** run the cells from top to bottom (`Shift+Enter`). Everything runs on the
# free Colab CPU in about 5 minutes (mostly network training); no GPU and no simulator installation is
# needed – the simulations were done in advance with COMSOL by the teacher.

# %% [markdown]
# ## 1 Motivation: why a neural network for a transistor?
#
# A circuit simulator (SPICE) evaluates the current of every transistor millions of times during one
# simulation. It therefore cannot solve the semiconductor equations in the device (that takes seconds per
# bias point, see Section 3); instead it uses a **compact model** – a set of closed-form equations with tens
# to hundreds of fitted parameters (e.g. BSIM). Developing a compact model for a new device type takes years
# of expert work.
#
# A **neural network** trained on device simulations (or measurements) is an alternative: a generic function
# $I_D = f_\theta(\text{design}, V_G, V_D)$ that is as fast as a compact model and can be retrained for a new
# technology in minutes. This is an active research topic in the semiconductor industry. The open questions
# are exactly the ones you will meet here: How much data is needed? How accurate are the *derivatives*
# (circuit simulators need $\partial I_D/\partial V_G$ and $\partial I_D/\partial V_D$)? Does the network obey
# physics outside the training points?

# %% [markdown]
# ## 2 Physics of the MOSFET
#
# We study an **n-channel MOSFET**: a p-type silicon substrate (acceptor concentration $N_A$) with two heavily
# doped n⁺ regions, the **source** and the **drain**. Between them, a **gate** electrode is separated from the
# silicon by a thin oxide of thickness $t_{ox}$. The gate length is $L_g$.
#
# ```
#                  gate (V_G)
#                ┌───────────┐
#   source (0 V) │   oxide   │ drain (V_D)
#   ──┬────────┬─┴───────────┴─┬────────┬──
#     │   n+   │   channel     │   n+   │      ← junction depth x_j
#     └────────┘  p-type, N_A  └────────┘
#                   substrate (0 V)
# ```
#
# **How it works.** A positive gate voltage pushes holes away from the surface and attracts electrons. When
# $V_G$ exceeds the **threshold voltage** $V_{th}$, a thin layer of electrons (the *inversion layer*, or
# channel) connects source and drain, and a current flows when $V_D>0$.
#
# ### Threshold voltage
# The classical long-channel result is
# $$V_{th} = V_{FB} + 2\phi_F + \frac{\sqrt{2\,\varepsilon_{Si}\,q\,N_A\,(2\phi_F)}}{C_{ox}},
# \qquad C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}},\qquad \phi_F = \frac{k_BT}{q}\ln\frac{N_A}{n_i},$$
# * $2\phi_F$ – band bending needed to make the surface as n-type as the bulk is p-type (*strong inversion*),
# * the last term – voltage across the oxide needed to hold the charge of the depleted acceptors,
# * $V_{FB} = \Phi_M - \Phi_S$ – the **flat-band voltage**, the difference of the work functions of the gate
#   metal and of the p-type silicon, $\Phi_S = \chi + E_g/q - (k_BT/q)\ln(N_v/N_A)$.
#
# $V_{th}$ grows with doping $N_A$ and oxide thickness $t_{ox}$, and decreases with temperature (because $n_i$
# grows).
#
# ### Drain current (square-law model)
# Above threshold the channel charge per unit area is $C_{ox}(V_G - V_{th} - V(x))$, where $V(x)$ is the
# channel potential. Integrating the drift current along the channel gives, for channel width $W$,
# $$I_D = \mu_n C_{ox}\frac{W}{L}\left[(V_G-V_{th})V_D - \frac{V_D^2}{2}\right]\quad (V_D < V_G-V_{th},\ \text{linear region}),$$
# $$I_D = \frac{\mu_n C_{ox}}{2}\frac{W}{L}(V_G-V_{th})^2\quad (V_D \ge V_G-V_{th},\ \text{saturation}).$$
#
# ### Subthreshold region
# Below threshold there is no channel, only a small **diffusion** current of electrons that grows exponentially
# with $V_G$:
# $$I_D \propto \exp\!\left(\frac{q V_G}{m\,k_BT}\right), \qquad
# S = \frac{dV_G}{d\log_{10}I_D} = \ln 10\,\frac{k_BT}{q}\,m,\qquad m = 1 + \frac{C_d}{C_{ox}},$$
# where $C_d = \varepsilon_{Si}/W_d$ is the capacitance of the depletion layer of width
# $W_d = \sqrt{2\varepsilon_{Si}\,2\phi_F/(qN_A)}$. The **subthreshold swing** $S$ is at least
# 60 mV/decade at room temperature – a fundamental limit for switching a MOSFET off.
#
# These formulas neglect many effects (2D electrostatics of short channels, the drop of mobility at the surface
# and at high fields, series resistance, …). A device simulator does not need such approximations.

# %% [markdown]
# ## 3 The COMSOL model and how the data were generated
#
# The data were computed with **COMSOL Multiphysics 6.3** (Semiconductor Module), starting from the
# Application Library model *DC Characteristics of a MOS Transistor*:
#
# * 2D cross-section of the transistor, 3 µm × 0.7 µm; the third dimension (channel width) is $W = 1\ \mu$m, so
#   **all currents are in A per µm of channel width**;
# * drift–diffusion equations (Poisson + electron and hole continuity, as in case study 1), **Fermi–Dirac**
#   statistics, band-gap narrowing (Jain–Roulston), SRH recombination;
# * **constant mobilities** $\mu_n = 1450$, $\mu_p = 500\ \mathrm{cm^2/(V\,s)}$ (the model does not include any
#   doping, temperature, surface or high-field dependence of the mobility – keep this in mind);
# * gate: *thin insulator gate* boundary condition (oxide not meshed), $\varepsilon_{ox}=4.5\,\varepsilon_0$,
#   metal work function $\Phi_M = 4.1$ V;
# * source, body and the bottom of the substrate at 0 V.
#
# **Numerics.** The equations are discretized with the finite-element method on a mesh refined under the gate
# (~5 000 elements, ~15 000 unknowns) and solved with Newton's method. The voltages are ramped in small steps
# (*continuation*): first $V_G$ at a small drain voltage, then $V_D$ for each $V_G$, each solution being the
# starting guess of the next. One device (51 bias points) took about **5 minutes** on 3 CPU cores.
#
# **The datasets**
#
# | file | content |
# |---|---|
# | `mosfet_tier1_grid.csv` | tier 1 device: 21 values of $V_G$ (0…5 V) × 16 values of $V_D$ (0…5 V) |
# | `mosfet_tier1_idvg.csv` | tier 1 device: $I_D$–$V_G$ at $V_D = 10$ mV, $V_G$ step 0.1 V |
# | `mosfet_tier2.csv` | 154 devices (Latin hypercube over the design parameters), 51 bias points each |
#
# The tier 1 device is the Application Library device: $N_A = 10^{17}\ \mathrm{cm^{-3}}$, $t_{ox} = 30$ nm,
# $L_g = 1.6\ \mu$m, $x_j = 0.25\ \mu$m, $T = 293.15$ K.

# %% [markdown]
# ## 4 Setup

# %%
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(1)
torch.manual_seed(1)

DATA = os.environ.get("PPSR_MOSFET_DIR", "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/"
                                         "comsol_datasets/data/mosfet")

# Physical constants and the material parameters of the COMSOL model
KB = 8.617333e-5          # Boltzmann constant (eV/K)
Q = 1.602177e-19          # elementary charge (C)
EPS0 = 8.854188e-14       # vacuum permittivity (F/cm)
EPS_SI, EPS_OX = 11.7 * EPS0, 4.5 * EPS0
EG, CHI, PHI_M = 1.12, 4.05, 4.1   # band gap (eV), electron affinity (eV), gate work function (eV)
MU_N = 1450.0             # electron mobility (cm²/(V s))
W_CM = 1e-4               # channel width 1 µm in cm


def nc_nv(T):
    """Effective densities of states (cm⁻³), as in the COMSOL silicon material."""
    return 2.8e19 * (T / 300) ** 1.5, 1.04e19 * (T / 300) ** 1.5


def vth_analytic(NA, tox_nm, T):
    """Long-channel threshold voltage (V) from Section 2."""
    kT = KB * T
    Nc, Nv = nc_nv(T)
    ni = np.sqrt(Nc * Nv) * np.exp(-EG / (2 * kT))
    two_phi_f = 2 * kT * np.log(NA / ni)
    v_fb = PHI_M - (CHI + EG - kT * np.log(Nv / NA))
    cox = EPS_OX / (tox_nm * 1e-7)
    return v_fb + two_phi_f + np.sqrt(2 * EPS_SI * Q * NA * two_phi_f) / cox


def swing_analytic(NA, tox_nm, T):
    """Subthreshold swing (V/decade) from Section 2."""
    kT = KB * T
    Nc, Nv = nc_nv(T)
    ni = np.sqrt(Nc * Nv) * np.exp(-EG / (2 * kT))
    Wd = np.sqrt(2 * EPS_SI * 2 * kT * np.log(NA / ni) / (Q * NA))
    return np.log(10) * kT * (1 + (EPS_SI / Wd) / (EPS_OX / (tox_nm * 1e-7)))


print(f"tier 1 device: V_th = {vth_analytic(1e17, 30, 293.15):.3f} V, "
      f"S = {1e3 * swing_analytic(1e17, 30, 293.15):.0f} mV/decade (analytic)")

# %% [markdown]
# ## 5 Tier 1: one transistor
#
# ### 5.1 The data

# %%
grid = pd.read_csv(f"{DATA}/mosfet_tier1_grid.csv")
idvg = pd.read_csv(f"{DATA}/mosfet_tier1_idvg.csv")
print(f"grid: {len(grid)} bias points, Id–Vg curve: {len(idvg)} points")
grid.head()

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for vg in [1.5, 2.0, 3.0, 4.0, 5.0]:
    g = grid[grid.Vg_V == vg]
    ax1.plot(g.Vd_V, 1e6 * g.Id_A_per_um, "o-", ms=3, label=f"$V_G$ = {vg} V")
ax1.set(xlabel="$V_D$ (V)", ylabel="$I_D$ (µA/µm)", title="Output characteristics")
ax1.legend(fontsize=8)
ax2.semilogy(idvg.Vg_V, np.abs(idvg.Id_A_per_um), "o-", ms=3)
ax2.set(xlabel="$V_G$ (V)", ylabel="$|I_D|$ (A/µm)", title="Transfer characteristic, $V_D$ = 10 mV")
fig.tight_layout()

# %% [markdown]
# **Observe:** the linear and saturation regions in the output characteristics, and the exponential
# (straight-line) subthreshold region in the transfer characteristic. Below about $10^{-12}$ A/µm the current is
# numerical noise of the simulation (it can even be negative – we plot $|I_D|$), the same resolution limit as in
# the pn diode case study.
#
# ### 5.2 Threshold voltage and subthreshold swing from the simulation
#
# Standard extraction methods:
# * **$V_{th}$ by linear extrapolation:** at the point of maximum transconductance $g_m = dI_D/dV_G$ draw the
#   tangent to $I_D(V_G)$; its intersection with $I_D=0$, minus $V_D/2$, is $V_{th}$.
# * **$S$** from the slope of $\log_{10} I_D$ in the subthreshold region.

# %%
def extract_vth(vg, id_, vd):
    """Threshold voltage by linear extrapolation at maximum transconductance."""
    gm = np.gradient(id_, vg)
    k = np.argmax(gm)
    return vg[k] - id_[k] / gm[k] - vd / 2


def extract_swing(vg, id_, lo=1e-11, hi=1e-8):
    """Subthreshold swing (V/decade) from a straight-line fit of log10(Id) between lo and hi."""
    sel = (id_ > lo) & (id_ < hi)
    return 1 / np.polyfit(vg[sel], np.log10(id_[sel]), 1)[0]


vth_sim = extract_vth(idvg.Vg_V.values, idvg.Id_A_per_um.values, 0.01)
s_sim = extract_swing(idvg.Vg_V.values, idvg.Id_A_per_um.values)
print(f"V_th: COMSOL {vth_sim:.3f} V, analytic {vth_analytic(1e17, 30, 293.15):.3f} V")
print(f"S:    COMSOL {1e3 * s_sim:.0f} mV/decade, analytic {1e3 * swing_analytic(1e17, 30, 293.15):.0f} mV/decade")

# %% [markdown]
# The simple formulas of Section 2 describe this long-channel device well. (How many points does the swing fit
# actually use? Print `idvg` and check – the subthreshold region is only a few tenths of a volt wide.)
#
# ### 5.3 The textbook model as a compact model
#
# The square-law model has only two parameters: $V_{th}$ and $k = \mu_n C_{ox} W/L$. We fit them to the
# simulated points above threshold ($V_G \ge 1.5$ V) by least squares.

# %%
def square_law(v, vth, k):
    vg, vd = v
    vov = np.maximum(vg - vth, 0)  # overdrive voltage
    vde = np.minimum(vd, vov)      # drain voltage, clamped at the saturation voltage
    return k * (vov * vde - vde ** 2 / 2)


above = grid[(grid.Vg_V >= 1.5) & (grid.Vd_V > 0)]
(vth_fit, k_fit), _ = curve_fit(square_law, (above.Vg_V, above.Vd_V), above.Id_A_per_um, p0=(1.0, 1e-5))
rel_sq = np.abs(square_law((above.Vg_V, above.Vd_V), vth_fit, k_fit) / above.Id_A_per_um - 1)
print(f"fitted V_th = {vth_fit:.3f} V, k = {k_fit:.3e} A/V²  "
      f"(µn·Cox·W/Lg = {MU_N * EPS_OX / 30e-7 * 1e-4 / 1.6e-4:.3e} A/V²)")
print(f"square-law model: median rel. error {np.median(rel_sq):.1%}, max {rel_sq.max():.1%}")

# %% [markdown]
# Two parameters already give a rough description, but with errors of tens of percent – the simple model misses,
# e.g., the variation of the depletion charge along the channel (the *body effect*), which lowers the saturation
# current. Real compact models add many more parameters to fix this. A neural network takes the opposite route:
# a generic function with many parameters, all fitted to data.
#
# ### 5.4 A neural network for one transistor
#
# * **Inputs:** $V_G$, $V_D$ (standardized).
# * **Output:** the current spans 8 orders of magnitude, so – as in the pn diode case study – we use a
#   logarithm-like transformation. But there is a catch: for $V_D \to 0$ the current goes to zero
#   proportionally to $V_D$, so $\log I_D \to -\infty$ – a singularity no network can fit. We therefore factor out
#   the known dependence on small $V_D$ and let the network learn only the rest:
#   $$I_D = I_s\,\sinh(y)\,\left(1 - e^{-qV_D/k_BT}\right),\qquad I_s = 10^{-11}\ \mathrm{A/µm}.$$
#   The factor in brackets is exact for the diffusion current below threshold and goes to zero linearly
#   for small $V_D$ in all regions; for $V_D \gg k_BT/q$ it is 1, and $y \approx \ln(2I_D/I_s)$. The network
#   output $y$ is now a smooth function, and $I_D(V_D=0) = 0$ holds exactly, by construction.
#   This is *physics-informed output design*.
# * **Data cleaning:** at $V_D = 0$ the simulation gives noise of up to $10^{-8}$ A/µm instead of 0 (see the
#   first row of the grid in Section 5.1). We drop these rows from training – the output formula already gives
#   $I_D = 0$ there. *Always look at your training data before trusting it.*
# * **Training and test data:** the grid and the $I_D$–$V_G$ curve at $V_D = 10$ mV together. We hold out as
#   **test data** two complete columns of the grid ($V_D = 0.75$ V and $2.5$ V) and every second point of the
#   10 mV curve ($V_G = 0.1, 0.3, \dots$ V). The network never sees these drain voltages or gate voltages, so
#   the test measures how well it *interpolates*. The remaining points are split 85 % training / 15 % validation.

# %%
I_SCALE = 1e-11  # A/µm, scale of the asinh target transformation
I_MIN = 1e-10    # A/µm, smallest current used in the error statistics (well above the numerical noise)


T_TIER1 = 293.15  # K, temperature of the tier 1 device (the tier 1 files have no T_K column)


def clean(df):
    """Drops the V_D = 0 rows (numerical noise; the output formula gives I_D = 0 there exactly)."""
    return df[df.Vd_V > 0].reset_index(drop=True)


def drain_factor(df):
    T = df.T_K.values if "T_K" in df else T_TIER1
    return 1 - np.exp(-df.Vd_V.values / (KB * T))


def to_target(df):
    return np.arcsinh(df.Id_A_per_um.values / (I_SCALE * drain_factor(df))).astype(np.float32)


def from_target(y, df):
    return np.sinh(y) * I_SCALE * drain_factor(df)


class Surrogate(torch.nn.Module):
    """Fully connected network; `features` turns a data frame into the input matrix."""

    def __init__(self, features, x_train, y_train, width, depth):
        super().__init__()
        self.features = features
        self.register_buffer("x_mean", torch.tensor(x_train.mean(0)))
        self.register_buffer("x_std", torch.tensor(x_train.std(0) + 1e-6))
        self.y_mean, self.y_std = float(y_train.mean()), float(y_train.std())  # output scaling
        layers, n_in = [], x_train.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.SiLU()]
            n_in = width
        layers.append(torch.nn.Linear(n_in, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std).squeeze(-1) * self.y_std + self.y_mean

    def predict(self, df):
        """Drain current (A/µm) for the rows of a data frame."""
        with torch.no_grad():
            return from_target(self(torch.tensor(self.features(df))).numpy(), df)


def train_surrogate(features, train_df, val_df, width=64, depth=3, epochs=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    x_tr, y_tr = torch.tensor(features(train_df)), torch.tensor(to_target(train_df))
    x_va, y_va = torch.tensor(features(val_df)), torch.tensor(to_target(val_df))
    model = Surrogate(features, x_tr.numpy(), y_tr.numpy(), width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best, best_state = [], np.inf, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.mean((model(x_tr) - y_tr) ** 2)  # full-batch gradient step
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(x_va) - y_va) ** 2).item()
        history.append((loss.item(), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1}: train MSE {loss.item():.4f}, validation MSE {val:.4f}")
    model.load_state_dict(best_state)
    print(f"training time {time.time() - t0:.0f} s, best validation MSE {best:.4f}")
    return model, np.array(history)


def bias_features(df):
    return np.column_stack([df.Vg_V, df.Vd_V]).astype(np.float32)


t1 = clean(pd.concat([grid, idvg], ignore_index=True))
odd_vg = np.round(t1.Vg_V * 10).astype(int) % 2 == 1
is_test = ((t1.Vd_V == 0.01) & odd_vg) | t1.Vd_V.isin([0.75, 2.5])
t1_test, t1_rest = t1[is_test], t1[~is_test]
val_idx = rng.choice(len(t1_rest), size=int(0.15 * len(t1_rest)), replace=False)
t1_val = t1_rest.iloc[val_idx]
t1_train = t1_rest.drop(t1_rest.index[val_idx])
print(f"tier 1: {len(t1_train)} training, {len(t1_val)} validation, {len(t1_test)} test points")
model_t1, hist_t1 = train_surrogate(bias_features, t1_train, t1_val)


# %%
def rel_error(I_pred, I_true):
    ok = np.abs(I_true) >= I_MIN
    return np.abs(I_pred[ok] / I_true[ok] - 1)


test_curve = t1_test[t1_test.Vd_V == 0.01]
test_cols = t1_test[(t1_test.Vd_V != 0.01) & (t1_test.Vg_V >= 1.5)]
for name, d in [("test points of the 10 mV curve", test_curve),
                ("held-out columns Vd = 0.75 and 2.5 V, Vg >= 1.5 V", test_cols)]:
    rel = rel_error(model_t1.predict(d), d.Id_A_per_um.values)
    print(f"DNN, {name}: median rel. error {np.median(rel):.1%}, max {rel.max():.1%}")
rel = rel_error(square_law((test_cols.Vg_V, test_cols.Vd_V), vth_fit, k_fit), test_cols.Id_A_per_um.values)
print(f"square law, held-out columns: median rel. error {np.median(rel):.1%}, max {rel.max():.1%}")

# %%
fine = pd.DataFrame({"Vd_V": np.tile(np.linspace(0, 5, 201), 5),
                     "Vg_V": np.repeat([1.5, 2.0, 3.0, 4.0, 5.0], 201)})
fine["Id_dnn"] = model_t1.predict(fine)
fine["Id_sq"] = square_law((fine.Vg_V, fine.Vd_V), vth_fit, k_fit)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for (vg, f), color in zip(fine.groupby("Vg_V"), plt.cm.viridis(np.linspace(0, 0.9, 5))):
    g = grid[grid.Vg_V == vg]
    axes[0].plot(g.Vd_V, 1e6 * g.Id_A_per_um, "o", ms=4, color=color, label=f"$V_G$ = {vg} V")
    axes[0].plot(f.Vd_V, 1e6 * f.Id_dnn, "-", color=color)
    axes[0].plot(f.Vd_V, 1e6 * f.Id_sq, ":", color=color)
    axes[1].plot(f.Vd_V, 1e6 * np.gradient(f.Id_dnn, f.Vd_V), "-", color=color)
axes[0].set(xlabel="$V_D$ (V)", ylabel="$I_D$ (µA/µm)",
            title="COMSOL (points), DNN (lines), square law (dotted)")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="$V_D$ (V)", ylabel="$g_d = dI_D/dV_D$ (µS/µm)", title="Output conductance of the DNN")

vg_fine = np.linspace(0, 5, 251)
pred = model_t1.predict(pd.DataFrame({"Vg_V": vg_fine, "Vd_V": 0.01}))
train_curve = t1_rest[t1_rest.Vd_V == 0.01]
axes[2].semilogy(train_curve.Vg_V, np.abs(train_curve.Id_A_per_um), "ko", ms=3, label="COMSOL, training")
axes[2].semilogy(test_curve.Vg_V, np.abs(test_curve.Id_A_per_um), "o", color="tab:orange", ms=3,
                 label="COMSOL, test")
axes[2].semilogy(vg_fine, np.abs(pred), "r-", label="DNN")
axes[2].set(xlabel="$V_G$ (V)", ylabel="$|I_D|$ (A/µm)", ylim=(1e-14, 1e-4), title="$I_D$–$V_G$, $V_D$ = 10 mV")
axes[2].legend()
fig.tight_layout()

# %% [markdown]
# **Things to notice:**
# * The network fits the output characteristics much better than the square-law model.
# * The **output conductance** $g_d$ is a derivative of the network. In saturation it should be small and
#   smooth; wiggles here would cause problems in a circuit simulator (e.g. in the gain of an amplifier
#   $\propto g_m/g_d$). Small errors of the function can be large errors of its derivative.
# * Below about $10^{-12}$ A/µm the network follows the numerical noise of the simulation – it cannot know
#   which part of the data is physics.
# * The 10 mV curve is essential: without it, the smallest drain voltage in the data would be 0.05 V, and the
#   network would have to *extrapolate* to 10 mV (see question 2).

# %% [markdown]
# ## 6 Tier 2: many transistor designs
#
# ### 6.1 The data
#
# 154 devices with design parameters drawn by Latin hypercube sampling:
#
# | column | meaning | range |
# |---|---|---|
# | `NA_cm3` | substrate doping | $10^{16}$–$3\times10^{17}\ \mathrm{cm^{-3}}$ (log scale) |
# | `tox_nm` | oxide thickness | 10–40 nm |
# | `Lg_um` | gate length | 0.8–2.0 µm |
# | `xj_um` | source/drain junction depth | 0.15–0.35 µm |
# | `T_K` | temperature | 273–373 K |
#
# For each device: an $I_D$–$V_G$ curve at $V_D = 10$ mV ($V_G$ = 0…5 V in 0.5 V steps, `sweep = idvg`) and a
# grid $V_G \in \{1,…,5\}$ V × $V_D \in \{0, 0.1, …, 5\}$ V (`sweep = grid`), 51 bias points in total.

# %%
t2 = pd.read_csv(f"{DATA}/mosfet_tier2.csv")
devices = pd.read_csv(f"{DATA}/mosfet_tier2_devices.csv")
print(f"{t2.device_id.nunique()} devices, {len(t2)} rows; status: {devices.status.value_counts().to_dict()}")
t2.head()

# %%
DESIGN = ["NA_cm3", "tox_nm", "Lg_um", "xj_um", "T_K"]
first = t2.groupby("device_id").first()
fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
for ax, col in zip(axes, DESIGN):
    ax.hist(np.log10(first[col]) if col == "NA_cm3" else first[col], bins=15)
    ax.set_xlabel("log10 NA_cm3" if col == "NA_cm3" else col)
fig.suptitle("Design parameters of the 154 tier 2 devices")
fig.tight_layout()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for dev in ["m0001", "m0002", "m0003", "m0004", "m0005"]:
    d = t2[t2.device_id == dev]
    p = d.iloc[0]
    lbl = f"NA={p.NA_cm3:.0e}, tox={p.tox_nm:.0f} nm, Lg={p.Lg_um:.2f} µm, T={p.T_K:.0f} K"
    s = d[d.sweep == "idvg"]
    ax1.semilogy(s.Vg_V, np.abs(s.Id_A_per_um), "o-", ms=3, label=lbl)
    s = d[(d.sweep == "grid") & (d.Vg_V == 5)]
    ax2.plot(s.Vd_V, 1e6 * s.Id_A_per_um, "o-", ms=3)
ax1.set(xlabel="$V_G$ (V)", ylabel="$|I_D|$ (A/µm)", title="$I_D$–$V_G$ at $V_D$ = 10 mV")
ax1.legend(fontsize=7)
ax2.set(xlabel="$V_D$ (V)", ylabel="$I_D$ (µA/µm)", title="$I_D$–$V_D$ at $V_G$ = 5 V")
fig.tight_layout()

# %% [markdown]
# The threshold voltage and the current level vary strongly between the designs. Notice also how coarse the
# $V_G$ sampling is: the subthreshold region is covered by only one or two points per device.
#
# ### 6.2 Two networks: plain inputs versus a physics-informed input
#
# * **Model P (plain):** inputs $\log_{10}N_A$, $t_{ox}$, $L_g$, $x_j$, $T/300\,\mathrm{K}$, $V_G$, $V_D$.
# * **Model F (physics-informed feature):** the same inputs plus the gate overdrive
#   $V_G - V_{th}^{\mathrm{analytic}}(N_A, t_{ox}, T)$ computed with the formula of Section 2. It adds no new
#   information, but it tells the network which combination of the inputs matters – as $qV/k_BT$ did in the
#   pn diode case study.
#
# Both networks have 4 hidden layers of 128 neurons and are trained the same way. The split into training,
# validation and test devices (70/15/15 %) is done **by device**: the test devices are designs the network has
# never seen.

# %%
def design_features(df):
    return np.column_stack([np.log10(df.NA_cm3), df.tox_nm, df.Lg_um, df.xj_um, df.T_K / 300,
                            df.Vg_V, df.Vd_V]).astype(np.float32)


def physics_features(df):
    overdrive = df.Vg_V - vth_analytic(df.NA_cm3, df.tox_nm, df.T_K)
    return np.column_stack([design_features(df), overdrive]).astype(np.float32)


def split_by_device(df, fractions=(0.7, 0.15, 0.15), seed=0):
    ids = np.random.default_rng(seed).permutation(df.device_id.unique())
    n1 = int(fractions[0] * len(ids))
    n2 = n1 + int(fractions[1] * len(ids))
    return [df[df.device_id.isin(p)].reset_index(drop=True) for p in (ids[:n1], ids[n1:n2], ids[n2:])]


t2_train, t2_val, t2_test = split_by_device(clean(t2))
print(f"devices: {t2_train.device_id.nunique()} train, {t2_val.device_id.nunique()} validation, "
      f"{t2_test.device_id.nunique()} test")
print("\nModel P (plain inputs)")
model_P, hist_P = train_surrogate(design_features, t2_train, t2_val, width=128, depth=4)
print("\nModel F (with the physics-informed overdrive input)")
model_F, hist_F = train_surrogate(physics_features, t2_train, t2_val, width=128, depth=4)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(hist_P[:, 1], label="model P, validation")
ax.semilogy(hist_F[:, 1], label="model F, validation")
ax.set(xlabel="epoch", ylabel="MSE of $y$", title="Training history")
ax.legend()

# %% [markdown]
# ### 6.3 Accuracy on the test devices
#
# Relative error of $I_D$ for $V_D > 0$ and $|I_D| \ge I_{min} = 10^{-10}$ A/µm, separately for
# weak inversion (around and below threshold) and strong inversion (on-state).

# %%
def errors(model, df):
    I = df.Id_A_per_um.values
    ok = (df.Vd_V.values > 0) & (np.abs(I) >= I_MIN)
    rel = np.abs(model.predict(df[ok]) / I[ok] - 1)
    strong = np.abs(I[ok]) >= 1e-7
    return {"weak inversion: median": np.median(rel[~strong]),
            "weak inversion: 95th percentile": np.percentile(rel[~strong], 95),
            "strong inversion: median": np.median(rel[strong]),
            "strong inversion: 95th percentile": np.percentile(rel[strong], 95)}


pd.DataFrame({"model P (plain)": errors(model_P, t2_test),
              "model F (physics feature)": errors(model_F, t2_test)}).T.style.format("{:.2%}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
pred = model_F.predict(t2_test)
mask = t2_test.Vd_V > 0
sc = axes[0].scatter(np.abs(t2_test.Id_A_per_um[mask]), np.abs(pred[mask]), c=t2_test.Vg_V[mask], s=6)
lims = [1e-14, 1e-2]
axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set(xscale="log", yscale="log", xlim=lims, ylim=lims, xlabel="$|I_D|$ COMSOL (A/µm)",
            ylabel="$|I_D|$ DNN (A/µm)", title="Model F, test devices")
fig.colorbar(sc, ax=axes[0], label="$V_G$ (V)")

vg_fine, vd_fine = np.linspace(0, 5, 101), np.linspace(0, 5, 101)
for dev, color in zip(t2_test.device_id.unique()[:3], ["tab:blue", "tab:green", "tab:red"]):
    d = t2_test[t2_test.device_id == dev]
    p = d.iloc[0]
    s = d[d.sweep == "idvg"]
    axes[1].semilogy(s.Vg_V, np.abs(s.Id_A_per_um), "o", color=color, label=dev)
    curve = pd.DataFrame({**{c: p[c] for c in DESIGN}, "Vg_V": vg_fine, "Vd_V": 0.01})
    axes[1].semilogy(vg_fine, np.abs(model_F.predict(curve)), "-", color=color)
    s = d[(d.sweep == "grid") & (d.Vg_V == 4)]
    axes[2].plot(s.Vd_V, 1e6 * s.Id_A_per_um, "o", color=color, label=dev)
    curve = pd.DataFrame({**{c: p[c] for c in DESIGN}, "Vg_V": 4.0, "Vd_V": vd_fine})
    axes[2].plot(vd_fine, 1e6 * model_F.predict(curve), "-", color=color)
axes[1].set(xlabel="$V_G$ (V)", ylabel="$|I_D|$ (A/µm)", ylim=(1e-14, 1e-3),
            title="Test devices: $I_D$–$V_G$, $V_D$ = 10 mV")
axes[2].set(xlabel="$V_D$ (V)", ylabel="$I_D$ (µA/µm)", title="Test devices: $I_D$–$V_D$, $V_G$ = 4 V")
axes[1].legend()
axes[2].legend()
fig.tight_layout()

# %% [markdown]
# ### 6.4 Did the networks learn the physics?
#
# With the network we can do something the dataset alone does not allow: change **one** design parameter
# while keeping all others fixed (in the dataset all parameters change at once). We compare the trends with
# the formulas of Section 2 for a reference design $t_{ox} = 25$ nm, $L_g = 1.4\ \mu$m, $x_j = 0.25\ \mu$m,
# $T = 300$ K, $N_A = 10^{17}\ \mathrm{cm^{-3}}$.
#
# **(a) Threshold voltage versus doping** for two oxide thicknesses, extracted from the network's $I_D$–$V_G$
# curves (at $V_D = 10$ mV, $V_G$ step 0.02 V) by the method of Section 5.2. For model F this check is partly
# circular (the analytic $V_{th}$ is one of its inputs); model P had to learn it from the data.
#
# **(b) Linear-region current versus gate length** at $V_G = 4$ V, $V_D = 0.1$ V, compared with the
# square-law formula with $L = L_g$ and the analytic $V_{th}$: $I_D \propto 1/L_g$.
#
# First: is the analytic formula good enough to serve as a reference? We extract $V_{th}$ from the (coarse,
# 0.5 V step) COMSOL $I_D$–$V_G$ curves of all 154 devices and compare.

# %%
diff = []
for dev, d in t2[t2.sweep == "idvg"].groupby("device_id"):
    p = d.iloc[0]
    diff.append(extract_vth(d.Vg_V.values, d.Id_A_per_um.values, 0.01) - vth_analytic(p.NA_cm3, p.tox_nm, p.T_K))
print(f"V_th(COMSOL) - V_th(analytic) over {len(diff)} devices: mean {np.mean(diff) * 1e3:.0f} mV, "
      f"standard deviation {np.std(diff) * 1e3:.0f} mV, max |difference| {np.max(np.abs(diff)) * 1e3:.0f} mV")

# %%
REF = {"NA_cm3": 1e17, "tox_nm": 25.0, "Lg_um": 1.4, "xj_um": 0.25, "T_K": 300.0}


def sweep(**changes):
    """Data frame with the reference design, some columns replaced by arrays (broadcast to equal length)."""
    cols = {**REF, **changes}
    n = max(np.size(v) for v in cols.values())
    return pd.DataFrame({c: np.broadcast_to(v, n) for c, v in cols.items()})


vg = np.arange(0, 5.001, 0.02)
na_values = np.logspace(16, np.log10(3e17), 15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for tox, color in [(15.0, "tab:blue"), (35.0, "tab:red")]:
    ax1.semilogx(na_values, vth_analytic(na_values, tox, 300.0), "-", color=color,
                 label=f"analytic, $t_{{ox}}$ = {tox:.0f} nm")
    for model, marker, name in [(model_P, "s", "P"), (model_F, "o", "F")]:
        vth_dnn = [extract_vth(vg, model.predict(sweep(NA_cm3=na, tox_nm=tox, Vg_V=vg, Vd_V=0.01)), 0.01)
                   for na in na_values]
        ax1.semilogx(na_values, vth_dnn, marker, color=color, mfc="none", ms=5, label=f"model {name}")
ax1.set(xlabel="$N_A$ (cm⁻³)", ylabel="$V_{th}$ (V)", title="(a) Threshold voltage")
ax1.legend(fontsize=7)

lg = np.linspace(0.8, 2.0, 25)
vth_ref = vth_analytic(REF["NA_cm3"], REF["tox_nm"], REF["T_K"])
k_ref = MU_N * EPS_OX / (REF["tox_nm"] * 1e-7) * W_CM / (lg * 1e-4)
ax2.plot(1 / lg, 1e6 * k_ref * ((4.0 - vth_ref) * 0.1 - 0.1 ** 2 / 2), "k-", label="square law")
for model, style, name in [(model_P, "s--", "P"), (model_F, "o--", "F")]:
    ax2.plot(1 / lg, 1e6 * model.predict(sweep(Lg_um=lg, Vg_V=4.0, Vd_V=0.1)), style, ms=4, mfc="none",
             label=f"model {name}")
ax2.set(xlabel="$1/L_g$ (1/µm)", ylabel="$I_D$ (µA/µm)", title="(b) Linear region, $V_G$ = 4 V, $V_D$ = 0.1 V")
ax2.legend()
fig.tight_layout()

# %% [markdown]
# **Things to notice:**
# * In (a) both networks reproduce the trend – $V_{th}$ rises with $N_A$ and $t_{ox}$ – but in places they
#   deviate from the formula by several tenths of a volt, although their current errors in Section 6.3 are
#   small. The COMSOL data typically agree with the formula within a few tens of mV (previous cell), so here the
#   **networks** are wrong, not the physics. $V_{th}$ is extracted from the *slope* of $I_D(V_G)$, and the
#   training data contain $V_G$ only in 0.5 V steps: the network is free to place the steep rise of the current
#   anywhere between two training points. Even model F, which gets the analytic $V_{th}$ as an input, does not
#   necessarily use it correctly.
# * In (b) the current is proportional to $1/L_g$, but larger than the square law, increasingly so for short
#   gates. Part of the reason: the source and drain regions extend under the gate, so the effective channel
#   is shorter than $L_g$.
# * A small average error is not enough. A surrogate used for design must be checked against the physical
#   trends the user cares about – and the training data must be dense where those trends are decided.

# %% [markdown]
# ## 7 Speed: simulator versus surrogate

# %%
n_dev = 1000
batch = pd.DataFrame({
    "NA_cm3": np.repeat(10 ** rng.uniform(16, np.log10(3e17), n_dev), 51),
    "tox_nm": np.repeat(rng.uniform(10, 40, n_dev), 51),
    "Lg_um": np.repeat(rng.uniform(0.8, 2.0, n_dev), 51),
    "xj_um": np.repeat(rng.uniform(0.15, 0.35, n_dev), 51),
    "T_K": np.repeat(rng.uniform(273, 373, n_dev), 51),
    "Vg_V": np.tile(rng.uniform(0, 5, 51), n_dev),
    "Vd_V": np.tile(rng.uniform(0, 5, 51), n_dev)})
t0 = time.time()
model_F.predict(batch)
t_dnn = (time.time() - t0) / n_dev
t_sim = devices[devices.status == "ok"].runtime_s.median()
print(f"COMSOL: {t_sim:.0f} s per device (51 bias points, 3 CPU cores), "
      f"DNN: {t_dnn * 1e6:.0f} µs per device → about {t_sim / t_dnn:,.0f}× faster")

# %% [markdown]
# Generating the training data took about 13 hours (154 devices × 5 minutes on 3 CPU cores). This is the real
# cost of a surrogate model: it pays off only if it is evaluated many times – in circuit simulation, in
# design optimization, or in statistical (Monte Carlo) analysis of manufacturing variations.

# %% [markdown]
# ## 8 Summary and questions
#
# * For a long-channel MOSFET the textbook formulas for $V_{th}$ and $S$ agree well with the simulation, but the
#   two-parameter square-law model describes the $I_D(V_G, V_D)$ characteristics only roughly.
# * A small neural network reproduces the characteristics of one transistor (tier 1) much more accurately, and
#   a larger one generalizes to new transistor designs (tier 2) with an error of a few percent in strong
#   inversion.
# * Derivatives (conductances), the subthreshold region and the physical trends must be checked separately –
#   a low average error does not guarantee them.
# * In the lecture, the same tier 2 dataset is used to train the built-in DNN of COMSOL Multiphysics; compare its
#   error with that of your models.
#
# **Questions**
# 1. Which of the input parameters influences the threshold voltage most? Use the formula in Section 2 and the
#    network (e.g. vary `T_K` in `sweep(...)`) to check.
# 2. In tier 1, train the network on the grid only (without the 10 mV curve) and evaluate it at $V_D = 10$ mV.
#    Above threshold the prediction is too large by about 60 %. Explain why, using the output formula of
#    Section 5.4 and the drain voltages present in the grid. (Hint: compare $1-e^{-qV_D/k_BT}$ with $V_D$ for
#    $V_D$ = 10 mV and 50 mV.)
# 3. Compare models P and F in Section 6.3. In which region does the physics-informed input help most, and why?
# 4. The COMSOL model uses a constant electron mobility. Which trend in Section 6.4 would change if the
#    mobility decreased with temperature, as it does in real silicon?
# 5. Evaluate model F for a gate length of 0.3 µm (outside the training range). Is the result plausible?
#    What would you expect physically for such a short channel?
