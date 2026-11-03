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
# # Case study 4: the bipolar transistor with self-heating – a neural network for an electrothermal device
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zemenja1/ppsr-ai-svs/blob/main/notebooks/03_bjt_thermal.ipynb)
#
# *Teaching materials of the project “AI ve výuce – Elektrotechnologie”, CTU in Prague, FEE.*
# *Version 1 (November 2026). License: text CC BY 4.0, code MIT.*
#
# **What you will do**
#
# 1. Recall how a bipolar transistor works, how its current depends on temperature, and how a transistor
#    heats itself up (*self-heating*).
# 2. **Explore the data:** Gummel plots and output characteristics of 400 transistor designs and operating
#    conditions simulated with COMSOL Multiphysics, each computed twice – at constant temperature
#    (*isothermal*) and with the semiconductor coupled to heat transfer (*self-heating*). Extract the thermal
#    resistance of the device from the data.
# 3. **Gummel network:** train a neural network that predicts the collector and base currents and the
#    temperature rise from the design, the cooling and the base voltage, and check whether it learned the
#    physics.
# 4. **Output-characteristics network:** predict $I_C(V_C)$ at a fixed base current, and use the network to
#    find a safe operating voltage for a given cooling.
#
# **How to work with this notebook:** run the cells from top to bottom (`Shift+Enter`). Everything runs on the
# free Colab CPU in about 10 minutes (mostly network training); no GPU and no simulator installation is
# needed – the simulations were done in advance with COMSOL by the teacher.

# %% [markdown]
# ## 1 Motivation: why self-heating, and why a neural network?
#
# A transistor converts part of the electrical power into heat, $P = I_C V_{CE} + I_B V_{BE}$. The heat must
# flow through the silicon, the package and the heat sink to the ambient, and on the way it raises the
# temperature of the device. In a power transistor or in a densely packed integrated circuit the temperature
# rise is tens of kelvin – and the current of a bipolar transistor at a fixed base–emitter voltage roughly
# **doubles every 10–15 K**. More current means more power and even more heat: an *electrothermal feedback
# loop*. If the loop gain exceeds one, the transistor destroys itself (**thermal runaway**).
#
# Circuit simulators therefore use **electrothermal compact models** (e.g. HICUM, VBIC, Mextram), which add a
# “thermal node” with a thermal resistance and capacitance to the electrical equations. As for the MOSFET in
# case study 3, developing and fitting such a model is expert work, while a neural network trained on device
# simulations can be generated automatically. Self-heating makes the task harder: the network must learn a
# *coupled* problem, in which the result depends on the cooling (the thermal resistance to the ambient) as
# much as on the transistor itself.

# %% [markdown]
# ## 2 Physics of the bipolar transistor
#
# We study a vertical **npn transistor**: a heavily doped n⁺ emitter, a thin p-type base (doping $N_B$), a
# lightly doped n-type epitaxial collector (doping $N_{epi}$) and a heavily doped n⁺ buried collector. The
# emitter is grounded; the base voltage $V_B = V_{BE}$ and the collector voltage $V_C = V_{CE}$ are applied
# (or, for the output characteristics, the base current $I_B$ is applied).
#
# ```
#        emitter (0 V)   base (V_B)          collector (V_C)
#        ┌───────┐       ┌────┐              ┌────┐
#   ─────┤  n+   ├───────┤ p+ ├──────────────┤ n+ ├────   surface
#        └───────┘                           │    │
#     p-type base, N_B  (thin layer under the emitter)
#     n-type epitaxial collector, N_epi       │    │
#     n+ buried collector ────────────────────┘    │
# ```
#
# **How it works.** The forward-biased base–emitter junction injects electrons from the emitter into the
# base. The base is thin, so almost all electrons diffuse across it without recombining and are swept into the
# collector by the field of the reverse-biased base–collector junction. A small base current – mostly holes
# injected from the base back into the emitter – controls a large collector current.
#
# ### Gummel plot: currents versus base–emitter voltage
# $$I_C = I_S\,\exp\!\left(\frac{qV_{BE}}{k_BT}\right),\qquad
# I_S = \frac{q A_E\, D_n\, n_i^2}{G_B},\qquad G_B = \int_{\text{base}} N_B\,dx \approx N_B W_B,$$
# $$I_B = \frac{I_C}{\beta},\qquad
# \beta \approx \frac{D_n\,G_E}{D_p\,G_B}\ \ (\text{current gain}),$$
# where $A_E$ is the emitter area, $W_B$ the base width and $G_B$, $G_E$ the **Gummel numbers** of the base
# and of the emitter. On a logarithmic current axis both currents are straight lines with a slope of
# $\ln 10\,k_BT/q \approx 60$ mV/decade. A more heavily doped base lowers $I_S$ and the current gain:
# $I_S,\ \beta \propto 1/N_B$.
#
# At **high currents** the simple picture fails:
# * *high injection* – the injected electron density in the base exceeds $N_B$, and the slope drops to
#   $\approx 120$ mV/decade;
# * *base push-out (Kirk effect)* – when the electron current density approaches $qN_{epi}v_{sat}$, the base
#   effectively extends into the lightly doped collector; the lower $N_{epi}$, the earlier the current gain
#   collapses;
# * *series resistances* of the emitter, base and collector – part of the applied $V_B$ drops across them,
#   so the internal junction voltage is smaller than $V_B$.
#
# ### Temperature dependence
# The intrinsic concentration grows steeply with temperature,
# $n_i^2 = N_cN_v\,e^{-E_g/k_BT} \propto T^3\,e^{-E_g/k_BT}$, so
# $$\ln I_S = \text{const} + 3\ln T - \frac{E_g}{k_BT}\quad\Rightarrow\quad
# \left.\frac{dV_{BE}}{dT}\right|_{I_C} = -\frac{E_g/q + 3k_BT/q - V_{BE}}{T} \approx -1.5\ \text{to}\ -2\ \text{mV/K}.$$
# (The temperature dependence of the diffusion coefficient $D_n$ is neglected here.) At a **fixed $V_{BE}$**
# the collector current grows by several percent per kelvin. At a **fixed $I_B$** it grows only as fast as
# the current gain $\beta(T)$, i.e. much more slowly.
#
# ### Output characteristics: collector current versus collector voltage at fixed base current
# * **Active region** ($V_{CE}\gtrsim 0.3$ V): $I_C \approx \beta I_B\,(1 + V_{CE}/V_A)$. The *Early voltage*
#   $V_A$ describes the narrowing of the base by the growing base–collector depletion layer.
# * **Saturation** ($V_{CE}\lesssim 0.2$ V): the base–collector junction is also forward-biased and injects
#   electrons back into the base; $I_C$ falls quickly and at $V_{CE}=0$ it is even **negative** (the
#   collector acts as a second emitter).
#
# ### Self-heating
# In steady state the temperature rise is set by the dissipated power and the **thermal resistance**
# $R_{th}$ (K/W) between the device and the ambient at temperature $T_0$:
# $$T - T_0 = R_{th}\,P,\qquad P = I_C V_{CE} + I_B V_{BE}.$$
# $R_{th}$ consists of the resistance of the silicon itself (the heat flows through micrometres of silicon to
# the contacts) and of the external cooling path. Combining it with the temperature dependence of $I_C$ gives
# the **electrothermal loop gain** at fixed $V_{BE}$,
# $$G = R_{th}\,V_{CE}\,\frac{\partial I_C}{\partial T},$$
# and the temperature rise is amplified by $1/(1-G)$. For $G\to 1$ the device runs away. Driving the base
# with a current (fixed $I_B$) instead of a voltage makes $\partial I_C/\partial T$, and therefore $G$, much
# smaller – one reason why power transistors are biased through resistors.

# %% [markdown]
# ## 3 The COMSOL model and how the data were generated
#
# The data were computed with **COMSOL Multiphysics 6.3** (Semiconductor Module and Heat Transfer Module),
# using the Application Library model *Bipolar Transistor with Self-Heating* unchanged except for five sample
# parameters:
#
# * 2D cross-section of the transistor, 2.5 µm wide; the third dimension (out-of-plane length) is 3 µm, and
#   **all currents are in A for this 3 µm long device**;
# * emitter $N_E = 2\times10^{20}$, buried collector $8\times10^{19}\ \mathrm{cm^{-3}}$;
#   drift–diffusion equations with incomplete ionization of the dopants;
# * *Heat Transfer in Solids* in the whole device, heated by Joule heating and recombination; the heat
#   leaves through the three metal contacts, whose heat transfer coefficient is chosen so that the total
#   **thermal resistance of the contacts to the ambient is $R$** (parameter `R_K_W`): small $R$ = good
#   cooling.
#
# **Numerics.** Finite elements, Newton's method, and continuation (each bias point starts from the previous
# solution). Each sample is computed in three steps, as in the library model:
# 1. initialization at $V_B=0$ with the collector voltage ramped to the sample value;
# 2. **isothermal** sweep of $V_B$ (heat transfer switched off, the device stays at $T_0$);
# 3. **self-heating** sweep of $V_B$ (semiconductor and heat transfer fully coupled).
#
# For the output characteristics, the base contact is switched to a **current** boundary condition and $V_C$
# is swept down to 0 V and up to 5 V from the sample's collector voltage, again first isothermally and then
# with self-heating. One sample took a few minutes (Gummel) or about 15 minutes (output characteristics) on
# 3 CPU cores.
#
# **Sample parameters** (Latin hypercube, 400 samples, random order; the library values in brackets):
#
# | column | meaning | range |
# |---|---|---|
# | `T0_K` | ambient temperature | 273–343 K (293.15 K) |
# | `R_K_W` | thermal resistance contacts → ambient | 30–300 K/W, log scale (100 K/W) |
# | `VC_V` / `VC0_V` | collector voltage of the Gummel sweep / start of the output sweeps | 1–5 V (3 V) |
# | `NB_cm3` | base doping | $6\times10^{17}$–$1.4\times10^{18}\ \mathrm{cm^{-3}}$, log scale ($9\times10^{17}$) |
# | `Nepi_cm3` | epitaxial collector doping | $10^{16}$–$4\times10^{16}\ \mathrm{cm^{-3}}$, log scale ($2\times10^{16}$) |
#
# **The datasets** (folder `comsol_datasets/data/bjt_thermal`)
#
# | file | content |
# |---|---|
# | `bjt_thermal_gummel.csv` | Gummel sweeps $V_B$ = 0.4, 0.5, …, 0.8, 0.85, …, 1.1 V at the sample's $V_C$; `sweep` = `isothermal` or `selfheating`; currents $I_E$, $I_B$, $I_C$ and the maximum temperature $T_{max}$ |
# | `bjt_thermal_output.csv` | output characteristics: $I_B$ = 10 µA, 100 µA, 1 mA; $V_C$ = 0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5 V and the sample's own `VC0_V`; $I_E$, $I_C$, $V_{BE}$, $T_{max}$ |
# | `bjt_thermal_samples.csv`, `bjt_thermal_output_samples.csv` | one row per sample: status (`ok`, `partial`, `failed`), number of points, runtime |
# | `bjt_thermal_gummel_finemesh.csv`, `bjt_thermal_devsim.csv`, `bjt_thermal_mesh_study.csv` | checks of the data quality: a finer mesh and an independent simulator (Section 6.4) |
#
# Currents are positive when they flow **into** the contact, so $I_E<0$ and $I_E + I_B + I_C = 0$.
# $T_{max}$ is the hottest point of the device (equal to $T_0$ in the isothermal sweeps).
#
# Some simulations did not converge (see the status files); in particular some output-characteristic curves
# at the highest base currents are missing. Missing points are simply absent from the data.

# %% [markdown]
# ## 4 Setup

# %%
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(1)
torch.manual_seed(1)

DATA = os.environ.get("PPSR_BJT_DIR", "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/"
                                      "comsol_datasets/data/bjt_thermal")

KB = 8.617333e-5   # Boltzmann constant (eV/K)
EG = 1.12          # band gap of silicon (eV)

gum_all = pd.read_csv(f"{DATA}/bjt_thermal_gummel.csv")
gum_samples = pd.read_csv(f"{DATA}/bjt_thermal_samples.csv")
out_all = pd.read_csv(f"{DATA}/bjt_thermal_output.csv")
out_samples = pd.read_csv(f"{DATA}/bjt_thermal_output_samples.csv")
print(f"Gummel: {gum_all.sample_id.nunique()} samples, {len(gum_all)} rows; "
      f"status {gum_samples.status.value_counts().to_dict()}")
print(f"output characteristics: {out_all.sample_id.nunique()} samples, {len(out_all)} rows; "
      f"status {out_samples.status.value_counts().to_dict()}")
gum_all.head()


# %%
def is_sh(df):
    """1 for the self-heating rows, 0 for the isothermal rows."""
    return (df.sweep.values == "selfheating").astype(np.float32)


def frame(base, **changes):
    """Data frame with the columns of `base`, some replaced by arrays (broadcast to equal length)."""
    cols = {**base, **changes}
    n = max(np.size(v) for v in cols.values())
    return pd.DataFrame({c: np.broadcast_to(v, n) for c, v in cols.items()})


# %% [markdown]
# ## 5 The data
#
# ### 5.1 Gummel plots
#
# First the data cleaning. As in the other case studies, the smallest currents are numerical noise of the
# simulation: at $V_B = 0.4$ V ($I_C \approx 10^{-11}$–$10^{-7}$ A) the three terminal currents often do not
# even add up to zero. We drop $V_B = 0.4$ V and check Kirchhoff's current law on the rest.

# %%
gum = gum_all[gum_all.VB_V >= 0.45].copy()
positive = (gum.IC_A > 0) & (gum.IB_A > 0)
gum = gum[positive].reset_index(drop=True)
kcl = np.abs(gum.IE_A + gum.IB_A + gum.IC_A) / np.abs(gum.IE_A)
print(f"{len(gum)} rows with V_B >= 0.5 V ({(~positive).sum()} rows with I_C or I_B <= 0 dropped)")
print(f"|I_E + I_B + I_C| / |I_E|: median {np.median(kcl):.1e}, max {kcl.max():.1e}")

GPARAMS = ["T0_K", "R_K_W", "VC_V", "NB_cm3", "Nepi_cm3"]
first = gum.groupby("sample_id")[GPARAMS].first()
fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
for ax, col in zip(axes, GPARAMS):
    logscale = col in ("R_K_W", "NB_cm3", "Nepi_cm3")
    ax.hist(np.log10(first[col]) if logscale else first[col], bins=15)
    ax.set_xlabel(f"log10 {col}" if logscale else col)
fig.suptitle(f"Parameters of the {len(first)} samples")
fig.tight_layout()

# %%
sh = gum[gum.sweep == "selfheating"]
heating = (sh.Tmax_K - sh.T0_K).groupby(sh.sample_id).max()
hot, cool = heating.idxmax(), heating.idxmin()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for sid, color in [(hot, "tab:red"), (cool, "tab:blue")]:
    d = gum[gum.sample_id == sid]
    p = d.iloc[0]
    label = f"{sid}: T0={p.T0_K:.0f} K, R={p.R_K_W:.0f} K/W, VC={p.VC_V:.1f} V"
    for sw, style in [("isothermal", "--"), ("selfheating", "-")]:
        s = d[d.sweep == sw]
        lbl = f"{label}, {sw}"
        axes[0].semilogy(s.VB_V, s.IC_A, style, marker="o", ms=3, color=color, label=lbl)
        axes[0].semilogy(s.VB_V, s.IB_A, style, marker="s", ms=3, color=color, mfc="none")
        axes[1].semilogx(s.IC_A, s.IC_A / s.IB_A, style, marker="o", ms=3, color=color)
        if sw == "selfheating":
            axes[2].plot(s.VB_V, s.Tmax_K - s.T0_K, "-o", ms=3, color=color, label=label)
axes[0].set(xlabel="$V_B$ (V)", ylabel="current (A)", title="Gummel plot: $I_C$ (●), $I_B$ (○)")
axes[0].legend(fontsize=7)
axes[1].set(xlabel="$I_C$ (A)", ylabel="$\\beta = I_C/I_B$", title="Current gain (dashed: isothermal)")
axes[2].set(xlabel="$V_B$ (V)", ylabel="$T_{max} - T_0$ (K)", title="Temperature rise")
axes[2].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# **Observe:**
# * At low $V_B$ both currents are straight lines in the Gummel plot; above about 0.85 V they bend down (high
#   injection, series resistances).
# * The current gain is roughly constant at low currents and collapses at high currents.
# * The two sweeps agree at low $V_B$, where almost no power is dissipated. At high $V_B$ the self-heating
#   current is larger: the device is hotter, and $I_S$ grows with temperature.
# * The two samples differ in the temperature rise by an order of magnitude. Which parameters cause the
#   difference?
#
# ### 5.2 How ideal is the transistor? Slope and saturation current
#
# In the ideal region ($V_B$ = 0.5–0.7 V) we fit $\ln I_C = \ln I_S + V_B/(n\,k_BT_0/q)$ for every isothermal
# sample. The **ideality factor** $n$ should be 1. Then we check how the saturation current $I_S$ depends on
# temperature and base doping by fitting the formula of Section 2 to all samples:
# $$\ln I_S - 3\ln T_0 = c + a\,\ln N_B + b\,\ln N_{epi} - \frac{E_g}{k_BT_0}.$$
# We expect $a\approx -1$, $b \approx 0$ (the collector doping should not matter at low currents) and
# $E_g$ close to the band gap of silicon.

# %%
iso = gum[gum.sweep == "isothermal"]
rows = []
for sid, d in iso.groupby("sample_id"):
    low = d[d.VB_V <= 0.701]
    at06 = d[np.isclose(d.VB_V, 0.6)]
    if len(low) < 3 or len(at06) == 0:
        continue
    p = d.iloc[0]
    vt = KB * p.T0_K
    slope = np.polyfit(low.VB_V, np.log(low.IC_A), 1)[0]
    rows.append({"sample_id": sid, "T0_K": p.T0_K, "NB_cm3": p.NB_cm3, "Nepi_cm3": p.Nepi_cm3,
                 "n": 1 / (slope * vt), "lnIS": np.log(at06.IC_A.iloc[0]) - 0.6 / vt,
                 "beta": at06.IC_A.iloc[0] / at06.IB_A.iloc[0]})
ideal = pd.DataFrame(rows)

X = np.column_stack([np.ones(len(ideal)), np.log(ideal.NB_cm3), np.log(ideal.Nepi_cm3), 1 / (KB * ideal.T0_K)])
coef = np.linalg.lstsq(X, ideal.lnIS - 3 * np.log(ideal.T0_K), rcond=None)[0]
print(f"ideality factor n: median {ideal.n.median():.3f}, range {ideal.n.min():.3f} – {ideal.n.max():.3f}")
print(f"fit of I_S: a (N_B exponent) = {coef[1]:.2f}, b (N_epi exponent) = {coef[2]:.2f}, "
      f"E_g = {-coef[3]:.3f} eV")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(ideal.n, bins=30)
ax1.set(xlabel="ideality factor $n$", ylabel="samples", title="Slope of the Gummel plot, 0.5–0.7 V")
sc = ax2.scatter(ideal.NB_cm3, ideal.beta, c=ideal.T0_K, s=12)
ax2.set(xscale="log", yscale="log", xlabel="$N_B$ (cm⁻³)", ylabel="β at $V_B$ = 0.6 V",
        title="Current gain versus base doping")
fig.colorbar(sc, ax=ax2, label="$T_0$ (K)")
fig.tight_layout()

# %% [markdown]
# The ideal-diode law holds well at low currents. The fitted “band gap” and the doping exponent need not be
# exactly the textbook values: the model contains band-gap narrowing in the heavily doped regions, incomplete
# ionization, and temperature-dependent mobilities (question 1).
#
# ### 5.3 The thermal resistance of the device
#
# From the self-heating sweeps we compute the dissipated power $P = I_C V_C + I_B V_B$ (the emitter is at 0 V)
# and the effective thermal resistance $R_{th} = (T_{max} - T_0)/P$, and plot it against the parameter $R$ (the
# thermal resistance of the contacts to the ambient). If all heat had to cross only $R$, we would get
# $R_{th} = R$.

# %%
hot_rows = sh[sh.VB_V >= 0.79].copy()
hot_rows["P_W"] = hot_rows.IC_A * hot_rows.VC_V + hot_rows.IB_A * hot_rows.VB_V
hot_rows["Rth"] = (hot_rows.Tmax_K - hot_rows.T0_K) / hot_rows.P_W
hot_rows = hot_rows[hot_rows.P_W > 1e-4]   # enough power for a meaningful ratio

fits = {}
fig, ax = plt.subplots(figsize=(7, 4.5))
for vb, color in [(0.9, "tab:blue"), (1.0, "tab:green"), (1.1, "tab:red")]:
    d = hot_rows[np.isclose(hot_rows.VB_V, vb)]
    slope, intercept = np.polyfit(d.R_K_W, d.Rth, 1)
    fits[vb] = (intercept, slope)
    ax.plot(d.R_K_W, d.Rth, "o", ms=3, color=color, alpha=0.5, label=f"$V_B$ = {vb} V")
    r = np.linspace(30, 300, 2)
    ax.plot(r, intercept + slope * r, "-", color=color)
    print(f"V_B = {vb} V: R_th = {intercept:.0f} K/W + {slope:.2f} × R")
ax.plot([30, 300], [30, 300], "k:", label="$R_{th} = R$")
ax.set(xlabel="$R$ (K/W)", ylabel="$(T_{max}-T_0)/P$ (K/W)", title="Effective thermal resistance")
ax.legend()
R_INT, R_SLOPE = fits[1.0]

# %% [markdown]
# The heating is several times larger than $R \cdot P$: the line does not start at zero. The intercept is
# the **thermal resistance of the silicon itself** – the heat is generated in a volume of a few µm³ and has to
# flow through the silicon to the contacts. A rough estimate: $L/(kA)$ with $L\approx 1\ \mu$m,
# $A\approx 2.5\ \mu\mathrm{m}\times3\ \mu\mathrm{m}$ and $k\approx 150$ W/(m·K) gives ~900 K/W – the same order
# of magnitude. For a micrometre-sized device the internal thermal resistance dominates even when the cooling
# is poor.
#
# Note also that the fitted lines are slightly different for different $V_B$: the heat sources move
# (Joule heat in the collector, recombination heat in the base), and the thermal conductivity of silicon
# decreases with temperature.
#
# ### 5.4 How strong is the electrothermal feedback?
#
# At fixed $V_B$ the self-heating current is larger than the isothermal one. Section 2 predicts
# $\ln(I_C^{sh}/I_C^{iso}) \approx \left(\frac{3}{T} + \frac{E_g - qV_{BE}}{k_BT^2}\right)\Delta T$.
# Let us compare.

# %%
pair = gum.pivot_table(index=["sample_id", "VB_V"], columns="sweep", values=["IC_A", "Tmax_K"]).dropna()
ratio = np.log(pair["IC_A"]["selfheating"] / pair["IC_A"]["isothermal"])
dT = pair["Tmax_K"]["selfheating"] - pair["Tmax_K"]["isothermal"]
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(dT, ratio, c=pair.index.get_level_values("VB_V"), s=6)
dt_line = np.linspace(0, dT.max(), 2)
for v in [0.8, 1.1]:
    ax.plot(dt_line, (3 / 300 + (EG - v) / (KB * 300 ** 2)) * dt_line, "--",
            label=f"formula, $V_{{BE}}$ = {v} V, T = 300 K")
ax.set(xlabel="$T_{max} - T_0$ (K)", ylabel="$\\ln(I_C^{sh}/I_C^{iso})$", ylim=(-0.05, 1.5),
       title="Increase of $I_C$ by self-heating at fixed $V_B$")
fig.colorbar(sc, ax=ax, label="$V_B$ (V)")
ax.legend()

# %% [markdown]
# The simulated increase is **far smaller** than the simple formula predicts – the electrothermal feedback in
# this device is weak. Question 2 asks you to explain why.

# %% [markdown]
# ## 6 A neural network for the Gummel characteristics
#
# ### 6.1 Inputs, outputs and two output designs
#
# * **Inputs:** $T_0/300$ K, $\log_{10}R$, $V_C$, $\log_{10}N_B$, $\log_{10}N_{epi}$, $V_B$ and a flag that is 1
#   for the self-heating sweep and 0 for the isothermal sweep. One network learns both: the flag switches the
#   heat transfer “on”.
# * **Outputs (three at once):** $I_C$, $I_B$ and the temperature rise $\Delta T = T_{max}-T_0$ (scaled by
#   100 K; for the isothermal rows the prediction is set to 0).
# * **Model P (plain output):** the network predicts $\ln I_C$ and $\ln I_B$.
# * **Model F (physics-informed output):** the network predicts $\ln I_C - qV_B/k_BT_0$ and
#   $\ln I_B - qV_B/k_BT_0$, i.e. we divide out the ideal exponential dependence of Section 2. In the ideal
#   region the network then only has to learn $\ln I_S$ and $\ln(I_S/\beta)$ – nearly constants.
#
# Both networks have 4 hidden layers of 128 neurons. Training, validation and test samples (70/15/15 %) are
# split **by sample**, so the test samples are designs and cooling conditions the network has never seen.

# %%
class Surrogate(torch.nn.Module):
    """Fully connected network with several outputs; `features` turns a data frame into the input matrix
    and `decode` turns the network output back into physical quantities."""

    def __init__(self, features, decode, x_train, y_train, width, depth):
        super().__init__()
        self.features, self.decode = features, decode
        self.register_buffer("x_mean", torch.tensor(x_train.mean(0)))
        self.register_buffer("x_std", torch.tensor(x_train.std(0) + 1e-6))
        self.register_buffer("y_mean", torch.tensor(y_train.mean(0)))
        self.register_buffer("y_std", torch.tensor(y_train.std(0) + 1e-6))
        layers, n_in = [], x_train.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.SiLU()]
            n_in = width
        layers.append(torch.nn.Linear(n_in, y_train.shape[1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean

    def predict(self, df):
        """Physical outputs (data frame) for the rows of `df`."""
        with torch.no_grad():
            y = self(torch.tensor(self.features(df))).numpy()
        return self.decode(y, df)


def train_surrogate(features, codec, train_df, val_df, width=128, depth=4, epochs=3000, lr=3e-3, seed=0):
    encode, decode = codec
    torch.manual_seed(seed)
    x_tr, y_tr = torch.tensor(features(train_df)), torch.tensor(encode(train_df))
    x_va, y_va = torch.tensor(features(val_df)), torch.tensor(encode(val_df))
    model = Surrogate(features, decode, x_tr.numpy(), y_tr.numpy(), width, depth)

    def mse(x, y):  # mean squared error of the standardized outputs
        return torch.mean(((model(x) - y) / model.y_std) ** 2)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best, best_state = [], np.inf, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = mse(x_tr, y_tr)  # full-batch gradient step
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = mse(x_va, y_va).item()
        history.append((loss.item(), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1}: train MSE {loss.item():.5f}, validation MSE {val:.5f}")
    model.load_state_dict(best_state)
    print(f"training time {time.time() - t0:.0f} s, best validation MSE {best:.5f}")
    return model, np.array(history)


def split_by_sample(df, fractions=(0.7, 0.15, 0.15), seed=0):
    ids = np.random.default_rng(seed).permutation(df.sample_id.unique())
    n1 = int(fractions[0] * len(ids))
    n2 = n1 + int(fractions[1] * len(ids))
    return [df[df.sample_id.isin(p)].reset_index(drop=True) for p in (ids[:n1], ids[n1:n2], ids[n2:])]


def gummel_features(df):
    return np.column_stack([df.T0_K / 300, np.log10(df.R_K_W), df.VC_V, np.log10(df.NB_cm3),
                            np.log10(df.Nepi_cm3), df.VB_V, is_sh(df)]).astype(np.float32)


def gummel_codec(physics):
    """(encode, decode) of the Gummel outputs; physics=True divides out exp(qV_B/kT0)."""
    def shift(df):
        return df.VB_V.values / (KB * df.T0_K.values) if physics else 0.0

    def encode(df):
        s = shift(df)
        return np.column_stack([np.log(df.IC_A.values) - s, np.log(df.IB_A.values) - s,
                                (df.Tmax_K.values - df.T0_K.values) / 100]).astype(np.float32)

    def decode(y, df):
        s = shift(df)
        return pd.DataFrame({"IC_A": np.exp(y[:, 0] + s), "IB_A": np.exp(y[:, 1] + s),
                             "dT_K": 100 * y[:, 2] * is_sh(df)}, index=df.index)
    return encode, decode


g_train, g_val, g_test = split_by_sample(gum)
print(f"samples: {g_train.sample_id.nunique()} train, {g_val.sample_id.nunique()} validation, "
      f"{g_test.sample_id.nunique()} test")
print("\nModel P (plain output)")
model_P, hist_P = train_surrogate(gummel_features, gummel_codec(False), g_train, g_val)
print("\nModel F (physics-informed output)")
model_F, hist_F = train_surrogate(gummel_features, gummel_codec(True), g_train, g_val)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(hist_P[:, 1], label="model P, validation")
ax.semilogy(hist_F[:, 1], label="model F, validation")
ax.set(xlabel="epoch", ylabel="MSE (standardized outputs)", title="Training history")
ax.legend()


# %% [markdown]
# ### 6.2 Accuracy on the test samples

# %%
def gummel_report(model, df):
    p = model.predict(df)
    rows = {}
    for sw in ["isothermal", "selfheating"]:
        m = (df.sweep == sw).values
        for col in ["IC_A", "IB_A"]:
            r = np.abs(p[col].values[m] / df[col].values[m] - 1)
            rows[f"{sw}: {col} rel. error"] = f"median {np.median(r):.2%}, 95th pct. {np.percentile(r, 95):.2%}"
    m = (df.sweep == "selfheating").values
    e = np.abs(p.dT_K.values[m] - (df.Tmax_K - df.T0_K).values[m])
    rows["selfheating: Tmax - T0 abs. error"] = (f"median {np.median(e):.2f} K, 95th pct. "
                                                 f"{np.percentile(e, 95):.2f} K, max {e.max():.1f} K")
    return pd.Series(rows)


pd.DataFrame({"model P (plain)": gummel_report(model_P, g_test),
              "model F (physics output)": gummel_report(model_F, g_test)})

# %% [markdown]
# A small error of the currents is not yet a proof that the network learned *self-heating*: the self-heating
# effect on $I_C$ is only a few to tens of percent. The real test is whether the network predicts the
# **difference** between the two sweeps. At $V_B = 1.1$ V we compare the true and predicted relative increase
# $I_C^{sh}/I_C^{iso} - 1$ for the test samples.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
top = g_test[np.isclose(g_test.VB_V, 1.1)].copy()
for model, color, name in [(model_P, "tab:orange", "P"), (model_F, "tab:green", "F")]:
    top["IC_pred"] = model.predict(top).IC_A.values
    piv = top.pivot_table(index="sample_id", columns="sweep", values=["IC_A", "IC_pred"]).dropna()
    true_inc = piv["IC_A"]["selfheating"] / piv["IC_A"]["isothermal"] - 1
    pred_inc = piv["IC_pred"]["selfheating"] / piv["IC_pred"]["isothermal"] - 1
    axes[0].plot(100 * true_inc, 100 * pred_inc, "o", ms=4, color=color, mfc="none", label=f"model {name}")
lim = [0, 100 * true_inc.max() * 1.1]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set(xlabel="true increase of $I_C$ (%)", ylabel="predicted increase (%)",
            title="Self-heating effect, $V_B$ = 1.1 V, test samples")
axes[0].legend()

vb_fine = np.linspace(0.5, 1.1, 61)
for sid, color in zip(g_test.sample_id.unique()[:3], ["tab:blue", "tab:green", "tab:red"]):
    d = g_test[g_test.sample_id == sid]
    base = {c: d.iloc[0][c] for c in GPARAMS}
    for sw, marker, style in [("isothermal", "o", "--"), ("selfheating", "s", "-")]:
        s = d[d.sweep == sw]
        pred = model_F.predict(frame(base, VB_V=vb_fine, sweep=sw))
        axes[1].semilogy(s.VB_V, s.IC_A, marker, color=color, ms=4, mfc="none")
        axes[1].semilogy(vb_fine, pred.IC_A, style, color=color, label=f"{sid}, {sw}")
        if sw == "selfheating":
            axes[2].plot(s.VB_V, s.Tmax_K - s.T0_K, marker, color=color, ms=4, mfc="none")
            axes[2].plot(vb_fine, pred.dT_K, style, color=color, label=sid)
axes[1].set(xlabel="$V_B$ (V)", ylabel="$I_C$ (A)", title="Model F, test samples (points: COMSOL)")
axes[1].legend(fontsize=7)
axes[2].set(xlabel="$V_B$ (V)", ylabel="$T_{max} - T_0$ (K)", title="Temperature rise, model F")
axes[2].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# ### 6.3 Did the network learn the physics?
#
# With the network we can change **one** parameter while keeping all others fixed – something the Latin
# hypercube data do not contain. Reference: the library transistor ($T_0$ = 300 K, $R$ = 100 K/W,
# $V_C$ = 3 V, $N_B = 9\times10^{17}$, $N_{epi} = 2\times10^{16}\ \mathrm{cm^{-3}}$).
#
# * **(a) Temperature coefficient of $V_{BE}$:** for each $T_0$ we find (isothermal) the base voltage that
#   gives the same collector current as at 300 K and $V_B$ = 0.7 V, and compare the slope with the formula of
#   Section 2.
# * **(b) Current gain versus base doping** at $V_B$ = 0.7 V: expected $\beta\propto N_B^{-1}$ (roughly).
# * **(c) Thermal resistance:** $(T_{max}-T_0)/P$ from the network versus $R$, compared with the straight line
#   fitted to the data in Section 5.3.

# %%
GREF = {"T0_K": 300.0, "R_K_W": 100.0, "VC_V": 3.0, "NB_cm3": 9e17, "Nepi_cm3": 2e16}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) V_BE at constant I_C versus temperature
vb_fine = np.linspace(0.5, 0.9, 401)
ic_target = model_F.predict(frame(GREF, VB_V=0.7, sweep="isothermal")).IC_A.values[0]
t0_values = np.linspace(273, 343, 8)
for model, marker, name in [(model_P, "s", "P"), (model_F, "o", "F")]:
    vbe = [np.interp(np.log(ic_target), np.log(model.predict(frame(GREF, T0_K=t, VB_V=vb_fine,
                                                                   sweep="isothermal")).IC_A.values), vb_fine)
           for t in t0_values]
    slope = np.polyfit(t0_values, vbe, 1)[0]
    axes[0].plot(t0_values, vbe, marker, mfc="none", label=f"model {name}: {1e3 * slope:.2f} mV/K")
formula = -(EG + 3 * KB * 300 - 0.7) / 300
axes[0].plot(t0_values, 0.7 + formula * (t0_values - 300), "k-", label=f"formula: {1e3 * formula:.2f} mV/K")
axes[0].set(xlabel="$T_0$ (K)", ylabel="$V_{BE}$ (V)", title=f"(a) $V_{{BE}}$ at $I_C$ = {ic_target:.2e} A")
axes[0].legend(fontsize=8)

# (b) current gain versus base doping
nb_values = np.logspace(np.log10(6e17), np.log10(1.4e18), 15)
for model, marker, name in [(model_P, "s", "P"), (model_F, "o", "F")]:
    p = model.predict(frame(GREF, NB_cm3=nb_values, VB_V=0.7, sweep="isothermal"))
    beta = p.IC_A / p.IB_A
    slope = np.polyfit(np.log(nb_values), np.log(beta), 1)[0]
    axes[1].loglog(nb_values, beta, marker, mfc="none", label=f"model {name}: β ∝ $N_B^{{{slope:.2f}}}$")
axes[1].loglog(ideal.NB_cm3, ideal.beta, ".", color="gray", alpha=0.4, label="data, all samples (0.6 V)")
axes[1].set(xlabel="$N_B$ (cm⁻³)", ylabel="β", title="(b) Current gain, $V_B$ = 0.7 V")
axes[1].legend(fontsize=8)

# (c) effective thermal resistance versus R
r_values = np.logspace(np.log10(30), np.log10(300), 15)
for model, marker, name in [(model_P, "s", "P"), (model_F, "o", "F")]:
    p = model.predict(frame(GREF, R_K_W=r_values, VB_V=1.0, sweep="selfheating"))
    power = p.IC_A * GREF["VC_V"] + p.IB_A * 1.0
    axes[2].plot(r_values, p.dT_K / power, marker, mfc="none", label=f"model {name}")
axes[2].plot(r_values, R_INT + R_SLOPE * r_values, "k-", label="fit to all data (Section 5.3)")
axes[2].set(xlabel="$R$ (K/W)", ylabel="$(T_{max}-T_0)/P$ (K/W)", title="(c) Thermal resistance, $V_B$ = 1.0 V")
axes[2].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# **Things to notice:**
# * Compare the slopes in (a) and (b) with the expectations, and the two models with each other. Where do
#   they disagree with the formulas – and is it the network or the formula that is wrong? (Look at the data
#   points in (b) and at the fit in Section 5.2.)
# * In (c) the network has to combine two things it learned separately: the current (which sets the power)
#   and the temperature rise. The data in Section 5.3 scatter around the fitted line because the other
#   parameters vary; the network, evaluated at one design, should give a smooth curve near the line.

# %% [markdown]
# ### 6.4 How good are the training data? Mesh convergence and an independent check
#
# The network reproduces the COMSOL data to a few percent – but **are the data right?** A finite-element
# solution depends on the mesh, and a surrogate can never be more accurate than the simulations it was trained
# on. The library model was meshed for a demonstration, not for accurate high-current results. To check it,
# three extra (small) datasets were computed:
#
# | file | content |
# |---|---|
# | `bjt_thermal_mesh_study.csv` | the library device ($T_0$ = 293.15 K, $R$ = 100 K/W, $V_C$ = 3 V) on 8 different meshes, plus the DEVSIM result |
# | `bjt_thermal_gummel_finemesh.csv` | the first 70 samples (`b0001`–`b0070`) recomputed with 4× more elements in the depth direction (48 000 instead of 12 000) |
# | `bjt_thermal_devsim.csv` | 23 samples recomputed with a **different simulator**, DEVSIM (finite volumes, same geometry and physics, own heat solver, mesh-converged) |
#
# The full dataset was *not* recomputed on the fine mesh: that would take about 20 h more compute time. This
# is a typical situation – the dataset you have is not the dataset you would like to have.

# %%
mesh = pd.read_csv(f"{DATA}/bjt_thermal_mesh_study.csv")
fine = pd.read_csv(f"{DATA}/bjt_thermal_gummel_finemesh.csv")
dev = pd.read_csv(f"{DATA}/bjt_thermal_devsim.csv")
print(f"fine mesh: {fine.sample_id.nunique()} samples, DEVSIM: {dev.sample_id.nunique()} samples, "
      f"both: {len(set(fine.sample_id) & set(dev.sample_id))}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ref = mesh[mesh.mesh == "devsim"].set_index(["sweep", "VB_V"])
labels = {"library": "library (12k)", "library_ratio10": "weaker grading (12k)",
          "lateral80": "2× lateral (24k)", "depth200": "2× depth (24k)", "uniform_x2": "uniform 2× (48k)",
          "depth400": "4× depth (48k) = fine mesh", "uniform_x4": "uniform 4× (192k)"}
for m, lab in labels.items():
    d = mesh[(mesh.mesh == m) & (mesh.sweep == "isothermal") & (mesh.VB_V >= 0.8)]
    err = d.IC_A.values / ref.loc[[("isothermal", v) for v in d.VB_V], "IC_A"].values - 1
    axes[0].plot(d.VB_V, 100 * err, "o-", ms=4, label=lab)
axes[0].axhline(0, color="k", lw=1)
axes[0].set(xlabel="$V_B$ (V)", ylabel="$I_C$ COMSOL / DEVSIM − 1 (%)",
            title="Library device, isothermal: error of each mesh")
axes[0].legend(fontsize=7)

# convergence of I_C(1.1 V) with the number of elements, for two ways of refining the mesh
for sw, marker in [("isothermal", "o"), ("selfheating", "s")]:
    for family, color, name in [(["library", "depth200", "depth400"], "tab:blue", "depth only"),
                                (["library", "uniform_x2", "uniform_x4"], "tab:orange", "uniform")]:
        d = mesh[mesh.mesh.isin(family) & (mesh.sweep == sw) & np.isclose(mesh.VB_V, 1.1)].sort_values("elements")
        axes[1].plot(d.elements, 1e3 * d.IC_A, marker + "-", color=color,
                     mfc="none" if sw == "isothermal" else color, label=f"{name} refinement, {sw}")
    axes[1].axhline(1e3 * ref.loc[(sw, 1.1), "IC_A"], color="k", ls="--" if sw == "isothermal" else "-",
                    lw=1, label=f"DEVSIM, {sw}")
axes[1].set(xscale="log", xlabel="number of mesh elements", ylabel="$I_C$ at $V_B$ = 1.1 V (mA)",
            title="Mesh convergence, library device")
axes[1].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# Refining the mesh **lowers** $I_C$ at high current and moves COMSOL towards DEVSIM. The depth direction
# (where the current flows from the emitter through the thin base into the collector) matters, the lateral
# direction does not. At low current ($V_B \le 0.85$ V) all meshes agree within about 1 %.
#
# Now the dataset samples: how far are the library-mesh data from the fine-mesh data and from DEVSIM, and
# how does that compare with the error of the network (model F), which was trained on the library-mesh data?

# %%
key = ["sample_id", "sweep", "VB_V"]
cmp = (gum.merge(fine[key + ["IC_A", "IB_A", "Tmax_K"]], on=key, suffixes=("", "_fine"))
          .merge(dev[key + ["IC_A", "IB_A", "Tmax_K"]].rename(columns=lambda c: c if c in key else c + "_dev"),
                 on=key, how="left"))
cmp["IC_net"] = model_F.predict(cmp).IC_A.values
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, sw in zip(axes[:2], ["isothermal", "selfheating"]):
    d = cmp[cmp.sweep == sw]
    ax.plot(d.VB_V + 0.006, 100 * (d.IC_A / d.IC_A_fine - 1), ".", color="tab:red", alpha=0.4,
            label="library mesh vs fine mesh (70 samples)")
    dd = d.dropna(subset=["IC_A_dev"])
    ax.plot(dd.VB_V - 0.006, 100 * (dd.IC_A / dd.IC_A_dev - 1), "o", ms=4, mfc="none", color="tab:red",
            label="library mesh vs DEVSIM")
    ax.plot(dd.VB_V, 100 * (dd.IC_A_fine / dd.IC_A_dev - 1), "s", ms=4, mfc="none", color="tab:blue",
            label="fine mesh vs DEVSIM")
    ax.plot(d.VB_V + 0.012, 100 * (d.IC_net / d.IC_A - 1), "x", ms=4, color="tab:green", alpha=0.5,
            label="network vs library-mesh data")
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="$V_B$ (V)", ylabel="relative difference of $I_C$ (%)", title=f"$I_C$, {sw}",
           xlim=(0.48, 1.13))
    ax.legend(fontsize=7)
d = cmp[(cmp.sweep == "selfheating") & (cmp.VB_V >= 0.9)].dropna(subset=["Tmax_K_dev"])
for col, marker, color, lab in [("Tmax_K", "o", "tab:red", "library mesh"),
                                ("Tmax_K_fine", "s", "tab:blue", "fine mesh")]:
    axes[2].plot(d.Tmax_K_dev - d.T0_K, d[col] - d.T0_K, marker, ms=4, mfc="none", color=color, label=lab)
lim = [0, (d.Tmax_K - d.T0_K).max() * 1.05]
axes[2].plot(lim, lim, "k--", lw=1)
axes[2].set(xlabel="DEVSIM $T_{max} - T_0$ (K)", ylabel="COMSOL $T_{max} - T_0$ (K)",
            title="Temperature rise, $V_B \\geq$ 0.9 V")
axes[2].legend(fontsize=8)
fig.tight_layout()

rows = {}
for vb in [0.8, 0.9, 1.0, 1.1]:
    d = cmp[np.isclose(cmp.VB_V, vb) & (cmp.sweep == "isothermal")]
    dd = d.dropna(subset=["IC_A_dev"])
    rows[f"V_B = {vb} V"] = {
        "network vs data": np.median(np.abs(d.IC_net / d.IC_A - 1)),
        "library vs fine mesh": np.median(np.abs(d.IC_A / d.IC_A_fine - 1)),
        "library mesh vs DEVSIM": np.median(np.abs(dd.IC_A / dd.IC_A_dev - 1)),
        "fine mesh vs DEVSIM": np.median(np.abs(dd.IC_A_fine / dd.IC_A_dev - 1))}
print("median |relative difference| of I_C, isothermal sweep")
pd.DataFrame(rows).T.style.format("{:.1%}")

# %% [markdown]
# **Things to notice:**
# * Up to $V_B \approx 0.85$ V all three sources agree within 1–2 %: there the dataset is good, and the
#   network error is the dominant error.
# * At high current the **data error** (library mesh vs fine mesh or DEVSIM: typically 10–20 %, up to ~50 %)
#   is several times larger than the **network error**. Improving the network further would only reproduce
#   the discretization error more faithfully. Samples at low temperature are the worst.
# * The error is *systematic* (always too much $I_C$) and *smooth*: the data look perfectly fine, are
#   monotonic and satisfy $I_E+I_B+I_C=0$. No check of the dataset alone would reveal it – only a
#   convergence study or an independent simulator does.
# * The fine mesh agrees with DEVSIM to a few percent in the isothermal sweep; with self-heating a few
#   percent more remain. The self-heating *gain* $I_C^{sh}/I_C^{iso}$ is about the same on both COMSOL meshes
#   but a bit smaller in DEVSIM – the two simulators differ slightly in the electrothermal model.

# %% [markdown]
# ## 7 A neural network for the output characteristics
#
# ### 7.1 The data
#
# The output characteristics are what a circuit designer sees: the transistor is driven by a base current and
# the collector voltage is varied. Each sample has (up to) 3 curves × 10 collector voltages × 2 sweeps.

# %%
out = out_all.copy()
active = out.VC_V >= 0.45
kcl = np.abs(out.IE_A + out.IB_A + out.IC_A) / np.abs(out.IE_A)
print(f"{out.sample_id.nunique()} samples, {len(out)} rows")
print(f"|I_E + I_B + I_C| / |I_E| for V_C >= 0.5 V: median {np.median(kcl[active]):.1e}, "
      f"max {kcl[active].max():.1e}")
print("rows per sweep and base current:")
print(out.groupby(["sweep", "IB_A"]).sample_id.count().unstack())

IB_VALUES = [1e-5, 1e-4, 1e-3]
sid = out_samples[out_samples.status == "ok"].sample_id.iloc[0]
d = out[out.sample_id == sid].sort_values("VC_V")
p = d.iloc[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ib, color in zip(IB_VALUES, ["tab:blue", "tab:green", "tab:red"]):
    for sw, style in [("isothermal", "--"), ("selfheating", "-")]:
        s = d[np.isclose(d.IB_A, ib) & (d.sweep == sw)]
        axes[0].plot(s.VC_V, 1e3 * s.IC_A, style, marker="o", ms=3, color=color,
                     label=f"$I_B$ = {1e6 * ib:.0f} µA, {sw}")
        axes[1].plot(s.VC_V, s.IC_A / ib, style, marker="o", ms=3, color=color)
        axes[2].plot(s.VC_V, s.Tmax_K - s.T0_K, style, marker="o", ms=3, color=color)
axes[0].set(xlabel="$V_C$ (V)", ylabel="$I_C$ (mA)",
            title=f"{sid}: T0={p.T0_K:.0f} K, R={p.R_K_W:.0f} K/W")
axes[0].legend(fontsize=7)
axes[1].set(xlabel="$V_C$ (V)", ylabel="$I_C/I_B$", title="Effective current gain")
axes[2].set(xlabel="$V_C$ (V)", ylabel="$T_{max}-T_0$ (K)", title="Temperature rise")
fig.tight_layout()

# %% [markdown]
# **Observe:** the saturation region below ~0.3 V (with $I_C < 0$ at $V_C = 0$), the slope of the curves in
# the active region (Early effect), the **drop of the current gain at high base current** (high injection,
# Kirk effect) and the extra rise of $I_C$ with $V_C$ when the device heats up. (At $V_C=0$ the self-heating
# temperature can be a few mK *below* $T_0$ – the recombination and Peltier terms can cool slightly. Not an
# error.)
#
# ### 7.2 The network
#
# * **Inputs:** $T_0/300$ K, $\log_{10}R$, $\log_{10}N_B$, $\log_{10}N_{epi}$, $\log_{10}I_B$, $V_C$ and the
#   self-heating flag. The column `VC0_V` is **not** an input: it was only the starting point of the
#   simulated sweep and should not affect the steady-state result (question 4).
# * **Outputs:** the effective current gain $I_C/I_B$ (scaled by 1/100), $V_{BE}$ and $\Delta T/100$ K.
#   Predicting $I_C/I_B$ instead of $I_C$ is again a physics-informed choice: the base current sets the
#   scale, and $I_C/I_B$ varies only between about −1 and a few hundred (it is negative in saturation).

# %%
OPARAMS = ["T0_K", "R_K_W", "NB_cm3", "Nepi_cm3"]


def output_features(df):
    return np.column_stack([df.T0_K / 300, np.log10(df.R_K_W), np.log10(df.NB_cm3), np.log10(df.Nepi_cm3),
                            np.log10(df.IB_A), df.VC_V, is_sh(df)]).astype(np.float32)


def output_encode(df):
    return np.column_stack([df.IC_A.values / df.IB_A.values / 100, df.VBE_V.values,
                            (df.Tmax_K.values - df.T0_K.values) / 100]).astype(np.float32)


def output_decode(y, df):
    return pd.DataFrame({"IC_A": 100 * y[:, 0] * df.IB_A.values, "VBE_V": y[:, 1],
                         "dT_K": 100 * y[:, 2] * is_sh(df)}, index=df.index)


o_train, o_val, o_test = split_by_sample(out)
print(f"samples: {o_train.sample_id.nunique()} train, {o_val.sample_id.nunique()} validation, "
      f"{o_test.sample_id.nunique()} test")
model_O, hist_O = train_surrogate(output_features, (output_encode, output_decode), o_train, o_val)

# %% [markdown]
# ### 7.3 Accuracy on the test samples

# %%
p = model_O.predict(o_test)
act = (o_test.VC_V >= 0.45).values
for sw in ["isothermal", "selfheating"]:
    m = (o_test.sweep == sw).values
    r = np.abs(p.IC_A.values[m & act] / o_test.IC_A.values[m & act] - 1)
    e = 1e3 * np.abs(p.VBE_V.values[m] - o_test.VBE_V.values[m])
    print(f"{sw}: I_C (V_C >= 0.5 V) median rel. error {np.median(r):.2%}, 95th pct. {np.percentile(r, 95):.2%};"
          f" V_BE median abs. error {np.median(e):.1f} mV")
m = (o_test.sweep == "selfheating").values
e = np.abs(p.dT_K.values[m] - (o_test.Tmax_K - o_test.T0_K).values[m])
print(f"selfheating: Tmax - T0 median abs. error {np.median(e):.2f} K, max {e.max():.1f} K")

counts = o_test.groupby("sample_id").size()
vc_fine = np.linspace(0, 5, 101)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for sid, color in zip(counts[counts == counts.max()].index[:2], ["tab:blue", "tab:red"]):
    d = o_test[o_test.sample_id == sid]
    base = {c: d.iloc[0][c] for c in OPARAMS}
    for ib in IB_VALUES:
        for sw, marker, style in [("isothermal", "o", "--"), ("selfheating", "s", "-")]:
            s = d[np.isclose(d.IB_A, ib) & (d.sweep == sw)]
            pred = model_O.predict(frame(base, IB_A=ib, VC_V=vc_fine, sweep=sw))
            axes[0].plot(s.VC_V, s.IC_A / ib, marker, ms=4, mfc="none", color=color)
            axes[0].plot(vc_fine, pred.IC_A / ib, style, color=color)
            axes[1].plot(s.VC_V, s.VBE_V, marker, ms=4, mfc="none", color=color)
            axes[1].plot(vc_fine, pred.VBE_V, style, color=color)
            if sw == "selfheating":
                axes[2].plot(s.VC_V, s.Tmax_K - s.T0_K, marker, ms=4, mfc="none", color=color)
                axes[2].plot(vc_fine, pred.dT_K, style, color=color, label=f"{sid}, $I_B$ = {1e6 * ib:.0f} µA")
axes[0].set(xlabel="$V_C$ (V)", ylabel="$I_C/I_B$", title="Test samples (points: COMSOL, lines: DNN)")
axes[1].set(xlabel="$V_C$ (V)", ylabel="$V_{BE}$ (V)", title="Base–emitter voltage")
axes[2].set(xlabel="$V_C$ (V)", ylabel="$T_{max}-T_0$ (K)", title="Temperature rise")
axes[2].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# Look carefully at the saturation region: the data have points only at 0, 0.1, 0.2 and 0.5 V, so the network
# has to guess the shape of the knee in between – exactly the problem of the coarse $V_G$ sampling near the
# threshold in the MOSFET case study.
#
# ### 7.4 Using the network: a safe operating voltage
#
# A surrogate is useful when it answers questions that would need many simulations. Here: **what is the
# highest collector voltage at which the hottest point of the transistor stays below 380 K**, for
# $I_B$ = 1 mA, $T_0$ = 300 K, as a function of the cooling $R$? With COMSOL each point of this curve would be a
# sweep of several minutes; the network gives the whole curve instantly.
#
# As a cross-check, we also evaluate the simple thermal model of Section 5.3,
# $T_{max} = T_0 + R_{th}(R)\,P$, with $R_{th}$ fitted to the Gummel data and $P$ from the network – two
# different datasets, so the agreement is a real test.

# %%
OREF = {"T0_K": 300.0, "R_K_W": 100.0, "NB_cm3": 9e17, "Nepi_cm3": 2e16}
T_LIMIT, IB_DESIGN = 380.0, 1e-3
vc_fine = np.linspace(0, 5, 251)
r_values = np.logspace(np.log10(30), np.log10(300), 12)
vc_max, vc_max_simple = [], []
for r in r_values:
    pred = model_O.predict(frame(OREF, R_K_W=r, IB_A=IB_DESIGN, VC_V=vc_fine, sweep="selfheating"))
    tmax = OREF["T0_K"] + pred.dT_K.values
    simple = OREF["T0_K"] + (R_INT + R_SLOPE * r) * (pred.IC_A.values * vc_fine + IB_DESIGN * pred.VBE_V.values)
    for t, result in [(tmax, vc_max), (simple, vc_max_simple)]:
        over = np.nonzero(t > T_LIMIT)[0]
        result.append(vc_fine[over[0] - 1] if len(over) else np.nan)  # nan: below the limit up to 5 V

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(r_values, vc_max, "o-", label="DNN temperature")
ax.plot(r_values, vc_max_simple, "s--", mfc="none", label="$R_{th}$ fit (Section 5.3) × DNN power")
ax.set(xscale="log", xlabel="$R$ (K/W)", ylabel="highest $V_C$ (V)",
       title=f"$T_{{max}} \\leq$ {T_LIMIT:.0f} K at $I_B$ = 1 mA, $T_0$ = 300 K")
ax.legend()

# %% [markdown]
# Before trusting such a result for a real design you would verify a few points of the curve with the
# simulator – the network is accurate only where it has seen data (here: $R$ = 30–300 K/W, $V_C \le 5$ V).

# %% [markdown]
# ## 8 Speed: simulator versus surrogate

# %%
n_s = 1000
batch = pd.DataFrame({
    "T0_K": np.repeat(rng.uniform(273, 343, n_s), 60),
    "R_K_W": np.repeat(10 ** rng.uniform(np.log10(30), np.log10(300), n_s), 60),
    "NB_cm3": np.repeat(10 ** rng.uniform(np.log10(6e17), np.log10(1.4e18), n_s), 60),
    "Nepi_cm3": np.repeat(10 ** rng.uniform(16, np.log10(4e16), n_s), 60),
    "IB_A": np.tile(np.repeat(IB_VALUES, 20), n_s),
    "VC_V": np.tile(np.linspace(0, 5, 10).repeat(2), 3 * n_s),
    "sweep": np.tile(["isothermal", "selfheating"], 30 * n_s)})
t0 = time.time()
model_O.predict(batch)
t_dnn = (time.time() - t0) / n_s
t_sim = out_samples[out_samples.status == "ok"].runtime_s.median()
print(f"COMSOL: {t_sim:.0f} s per sample (60 output points, 3 CPU cores), "
      f"DNN: {t_dnn * 1e6:.0f} µs per sample → about {t_sim / t_dnn:,.0f}× faster")
print(f"COMSOL time for all output-characteristic samples: {out_samples.runtime_s.sum() / 3600:.0f} h")

# %% [markdown]
# ## 9 Summary and questions
#
# * At low currents the simulated transistor follows the ideal Gummel law; at high currents high injection,
#   the Kirk effect and series resistances take over, and the current gain collapses.
# * For a micrometre-sized device the thermal resistance of the silicon itself is larger than that of a
#   reasonable cooling path, so self-heating cannot be removed by better cooling alone.
# * One network can learn the isothermal and the self-heating characteristics together, if it gets a flag
#   that tells which is which; the *difference* between them (the self-heating effect) is the hard part and
#   must be checked separately.
# * Physics-informed output transformations (dividing out $e^{qV_B/k_BT}$, predicting $I_C/I_B$) make the
#   learning problem easier – check in Section 6.2 by how much.
# * In the lecture, the same datasets are used to train the built-in DNN of COMSOL Multiphysics; compare its
#   error with that of your models.
#
# **Questions**
# 1. In Section 5.2 the fitted band gap and the exponent of $N_B$ differ from the textbook values. List the
#    effects in the COMSOL model that the formula $I_S \propto n_i^2/N_B$ ignores. Which of them changes the
#    effective band gap?
# 2. In Section 5.4 the self-heating increase of $I_C$ is much smaller than the formula predicts. Give at least
#    two reasons. (Hints: what limits the current at $V_B$ = 1.1 V? Is $T_{max}$ the temperature of the
#    base–emitter junction?) Estimate the electrothermal loop gain $G$ from the data for the hottest sample.
# 3. Why does the current gain at $I_B$ = 1 mA (Section 7.1) depend on $N_{epi}$? Use model O to plot
#    $I_C/I_B$ at $V_C$ = 3 V versus $N_{epi}$ for the three base currents and explain the result with the Kirk
#    effect.
# 4. `VC0_V` was not used as an input in Section 7. Check that this is justified: find two samples with
#    similar parameters but different `VC0_V` and compare their curves, or add `VC0_V` as an input and see
#    whether the test error changes.
# 5. Evaluate the networks outside the training range – $R$ = 1000 K/W, or $T_0$ = 400 K. Is the result
#    plausible? What would really happen to the transistor?
# 6. Train model F of Section 6 only on the isothermal data and evaluate it with the flag set to 1. What does
#    the network predict for the self-heating case, and why?
# 7. Section 6.4: train model F only on the 70 fine-mesh samples, and once more on the same 70 samples from
#    the library-mesh data. Compare both with DEVSIM. Is a small accurate dataset better than a large
#    inaccurate one? How could you combine the two? (Hint: learn the *correction* fine/library.)
# 8. Estimate the mesh-converged $I_C$ at $V_B$ = 1.1 V by Richardson extrapolation of the depth refinement
#    (12k, 24k, 48k elements) in `bjt_thermal_mesh_study.csv`. What order of convergence do you get, and how
#    close is the result to DEVSIM?
