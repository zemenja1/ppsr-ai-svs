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
# # Case study 1: the pn diode – from device physics to a neural-network surrogate
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zemenja1/ppsr-ai-svs/blob/main/notebooks/01_pn_diode.ipynb)
#
# *Teaching materials of the project “AI ve výuce – Elektrotechnologie”, CTU in Prague, FEE.*
# *Version 1 (October 2026). License: text CC BY 4.0, code MIT.*
#
# **What you will do**
#
# 1. Recall the physics of a silicon pn junction and the equations a device simulator solves.
# 2. Explore a dataset of 500 simulated diodes computed with **COMSOL Multiphysics** (provided by the teacher).
# 3. Simulate diodes yourself with the open-source simulator **DEVSIM** and check that it agrees with COMSOL.
# 4. Generate your own training dataset with DEVSIM.
# 5. Train a **deep neural network (DNN)** that predicts the diode current from the design parameters –
#    a *surrogate model* that is thousands of times faster than the simulator.
# 6. Compare networks trained on COMSOL data and on your DEVSIM data, and check whether the network
#    learned physically sensible behaviour.
#
# **How to work with this notebook:** run the cells from top to bottom (`Shift+Enter`). The whole
# notebook runs in about 15–20 minutes on the free Colab CPU (most of it is the DEVSIM simulations and the
# network training); no GPU is needed.

# %% [markdown]
# ## 1 Physics of the pn junction
#
# We study an abrupt silicon pn junction of length $5\,\mu\mathrm{m}$: a p-type region with acceptor
# concentration $N_A$ next to an n-type region with donor concentration $N_D$, with ohmic metal contacts at
# both ends. A voltage $V$ is applied to the p-side (anode); $V>0$ is forward bias.
#
# ```
#   anode (V) |   p-type, N_A   |   n-type, N_D   | cathode (0 V)
#             0               2.5 µm            5 µm
# ```
#
# ### Drift–diffusion model
# The state of the semiconductor is described by the electrostatic potential $\psi$ and the electron and hole
# concentrations $n$ and $p$. In steady state they satisfy three coupled nonlinear equations:
#
# * **Poisson equation** (charge creates the electric field $\mathbf{E}=-\nabla\psi$):
#   $$\nabla\cdot(\varepsilon\nabla\psi) = -q\,(p - n + N_D - N_A)$$
# * **Continuity equations** (carriers are conserved except for recombination $R$):
#   $$\nabla\cdot\mathbf{J}_n = q\,R, \qquad \nabla\cdot\mathbf{J}_p = -q\,R$$
# * **Drift–diffusion current densities**:
#   $$\mathbf{J}_n = q\mu_n n\,\mathbf{E} + qD_n\nabla n, \qquad \mathbf{J}_p = q\mu_p p\,\mathbf{E} - qD_p\nabla p,
#   \qquad D = \mu\,\frac{k_BT}{q}\ \text{(Einstein relation)}$$
#
# The total current density is $J = J_n + J_p$; in 1D steady state it is the same everywhere in the device.
#
# ### Material models
# * **Intrinsic carrier density** – grows extremely fast with temperature:
#   $$n_i(T) = \sqrt{N_c N_v}\,\exp\!\left(-\frac{E_g}{2k_BT}\right), \qquad N_{c,v}\propto T^{3/2},\quad E_g = 1.12\ \mathrm{eV}$$
# * **Shockley–Read–Hall (SRH) recombination** through defects with carrier lifetime $\tau$:
#   $$R = \frac{np - n_i^2}{\tau\,(n + n_i) + \tau\,(p + p_i)}$$
#   ($R>0$: recombination in forward bias, $R<0$: generation in reverse bias.)
# * **Mobility** $\mu_{n,p}(N_A+N_D,\,T)$ – decreases with doping (scattering on ionized dopants) and with
#   temperature (scattering on lattice vibrations). We use the same empirical formula as the COMSOL model.
#
# ### What to expect: the ideal diode
# For moderate forward bias the current follows the Shockley equation
# $$J = J_0\left[\exp\!\left(\frac{qV}{n_{\mathrm{id}}k_BT}\right)-1\right],$$
# with ideality factor $n_{\mathrm{id}}\approx 1$ when diffusion of minority carriers dominates and
# $n_{\mathrm{id}}\approx 2$ when recombination in the depletion region dominates (low bias, short lifetime).
# For our short diode (regions much shorter than the diffusion length) the saturation current density is
# $$J_0 \approx q\,n_i^2\left(\frac{D_n}{W_p N_A} + \frac{D_p}{W_n N_D}\right),$$
# so it scales with $n_i^2$, i.e. very strongly with temperature. In reverse bias the current saturates at a
# small value given mainly by **generation** in the depletion region, $J_{\mathrm{gen}}\approx q n_i W/(2\tau)$.
# At high forward bias, **high injection** and the **resistance** of the neutral regions bend the curve down.

# %% [markdown]
# ## 2 Numerical methods
#
# A device simulator turns the differential equations into a large system of nonlinear algebraic equations:
#
# 1. **Mesh** – the device is divided into small elements; the mesh is fine (1 nm) near the junction where the
#    concentrations change by many orders of magnitude, and coarser (20 nm) near the contacts.
# 2. **Finite-volume discretization with the Scharfetter–Gummel scheme** – the current between two mesh nodes is
#    computed from an exact solution of the 1D drift–diffusion equation along the edge. This keeps the
#    carrier concentrations positive and the discretization stable, which a naive finite-difference formula
#    would not. Both COMSOL (the “finite volume” study of its model) and DEVSIM use this scheme.
# 3. **Newton's method** solves the nonlinear system: each iteration solves a linear system with the Jacobian
#    matrix, until the update is negligible.
# 4. **Continuation (bias ramping)** – Newton's method needs a good starting guess. We first solve only the
#    Poisson equation in equilibrium, then add the carrier equations, and then increase the applied voltage in
#    small steps, using each solution as the starting point for the next one. If a step fails, it is halved.
#
# ### Surrogate model
# A simulation of one I–V curve takes seconds. A **surrogate model** is a function
# $$\hat J = f_\theta(N_A, N_D, \tau, T, V)$$
# with parameters $\theta$ fitted to simulated examples. Here $f_\theta$ is a small fully connected neural
# network; once trained it evaluates thousands of I–V curves in milliseconds. The price: it is only as good as
# its training data, and only inside the parameter range it has seen.

# %% [markdown]
# ## 3 Setup

# %%
# Install DEVSIM (semiconductor device simulator) and Intel MKL (the linear-algebra library DEVSIM needs).
# %pip install -q devsim mkl

# %%
import os, sys, time, contextlib
# DEVSIM (via Intel MKL) and PyTorch each load an OpenMP runtime; on some systems (e.g. Windows) the
# second copy aborts the program unless this is set. It is harmless for this notebook.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import qmc

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(1)
torch.manual_seed(1)

# %% [markdown]
# ## 4 The teacher's COMSOL dataset
#
# The dataset contains I–V curves of **500 diodes** simulated with COMSOL Multiphysics 6.3 (Semiconductor
# Module). The design parameters were sampled by *Latin hypercube sampling*, which spreads the samples evenly
# over the parameter space:
#
# | column | meaning | range |
# |---|---|---|
# | `Na_cm3`, `Nd_cm3` | acceptor / donor doping | $10^{15}$–$10^{18}\ \mathrm{cm^{-3}}$ (log scale) |
# | `tau_s` | carrier lifetime | $10^{-8}$–$10^{-5}$ s (log scale) |
# | `T_K` | temperature | 273–373 K |
# | `V_V` | applied voltage | −2 … +0.8 V (23 values) |
# | `J_A_cm2` | current density | A/cm² |

# %%
COMSOL_URL = ("https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/"
              "comsol_datasets/data/pn_diode/pn_diode_iv.csv")
comsol = pd.read_csv(os.environ.get("PPSR_DIODE_CSV", COMSOL_URL))
print(f"{comsol.device_id.nunique()} diodes, {len(comsol)} rows")
comsol.head()

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
first = comsol.groupby("device_id").first()
for ax, col, log in zip(axes, ["Na_cm3", "Nd_cm3", "tau_s", "T_K"], [True, True, True, False]):
    ax.hist(np.log10(first[col]) if log else first[col], bins=20, color="tab:blue")
    ax.set_xlabel(f"log10 {col}" if log else col)
fig.suptitle("Design parameters of the 500 COMSOL diodes")
fig.tight_layout()


# %%
def plot_iv(ax, df, label=None, style="o-", **kw):
    """Plots |J| versus V on a log scale; reverse-bias points are drawn with open markers."""
    g = df.sort_values("V_V")
    ax.semilogy(g.V_V, np.abs(g.J_A_cm2), style, label=label, ms=3, **kw)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("|J| (A/cm²)")


fig, ax = plt.subplots(figsize=(6.5, 4))
for dev in ["d0001", "d0002", "d0003", "d0004", "d0005"]:
    g = comsol[comsol.device_id == dev]
    p = g.iloc[0]
    plot_iv(ax, g, f"Na={p.Na_cm3:.0e}, Nd={p.Nd_cm3:.0e}, τ={p.tau_s:.0e} s, T={p.T_K:.0f} K")
ax.legend(fontsize=7)
ax.set_title("COMSOL I–V curves of five diodes")

# %% [markdown]
# **Observe:** the current changes by more than ten orders of magnitude. In forward bias it first grows
# exponentially (a straight line in the log plot), then bends down at high current. In reverse bias it is tiny
# and almost constant. Diodes with a short lifetime or a high temperature have a much larger reverse current.
#
# **Numerical resolution limit.** At the heavily doped contacts the current is the small difference of very
# large drift and diffusion currents of the majority carriers. In double-precision arithmetic such a difference
# cannot be resolved below roughly $10^{-9}$–$10^{-8}\ \mathrm{A/cm^2}$, so smaller currents in the dataset
# (some reverse-bias and near-zero-bias points) are numerical noise, not physics. We will keep this in mind
# when comparing simulators and when training the network.

# %% [markdown]
# ## 5 Simulating a diode with DEVSIM
#
# [DEVSIM](https://devsim.org) is an open-source device simulator. The function below builds the same diode
# as the COMSOL model: the same geometry, doping, $n_i(T)$, mobility formula, SRH recombination and ideal
# ohmic contacts. Read through it – every step corresponds to Section 2.

# %%
import devsim
from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact,
    CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact, GetContactBiasName)
from devsim.python_packages.model_create import CreateSolution, CreateNodeModel, CreateEdgeModel

Q = 1.602176634e-19      # elementary charge (C)
KB = 1.380649e-23        # Boltzmann constant (J/K)
EPS0 = 8.8541878128e-14  # vacuum permittivity (F/cm); DEVSIM works in cm
LENGTH = 5e-4            # device length 5 µm (in cm), junction in the middle


@contextlib.contextmanager
def quiet():
    """Silences DEVSIM's solver progress. Depending on the environment it is written to Python's
    sys.stdout (Jupyter, Colab) or directly to the process output (plain Python), so both are redirected."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def mobility(mu_lattice, a_i, b_i, T):
    """Doping- and temperature-dependent mobility (cm²/Vs) as an expression of Ntot = Na + Nd.
    Lattice scattering mu_l, ionized-impurity scattering mu_i, combined by an empirical formula."""
    mul = f"({mu_lattice}*({T}/300)^(-2.2))"
    mui = f"((({a_i})*({T})^1.5/Ntot)/(log(1+({b_i})*({T})^2/Ntot)-({b_i})*({T})^2/(Ntot+({b_i})*({T})^2)))"
    x = f"(6*{mul}/{mui})^0.5"
    return f"{mul}*(1.025/(1+({x}/1.68)^1.43)-0.025)"


_counter = [0]


def simulate_diode(Na, Nd, tau, T, voltages, h_junction=1e-7, h_contact=2e-6):
    """Current density J (A/cm²) of the abrupt Si pn diode at the given anode voltages (V).
    Na, Nd in cm^-3, tau in s, T in K; h_junction, h_contact = mesh spacing (cm)."""
    with quiet():
        return _simulate_diode(Na, Nd, tau, T, voltages, h_junction, h_contact)


def _simulate_diode(Na, Nd, tau, T, voltages, h_junction, h_contact):
    _counter[0] += 1
    dev, reg, mesh = f"diode{_counter[0]}", "Si", f"mesh{_counter[0]}"

    # --- mesh: fine at the junction, coarse at the contacts
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=h_contact, tag="anode")
    devsim.add_1d_mesh_line(mesh=mesh, pos=LENGTH / 2, ps=h_junction, tag="junction")
    devsim.add_1d_mesh_line(mesh=mesh, pos=LENGTH, ps=h_contact, tag="cathode")
    devsim.add_1d_contact(mesh=mesh, name="anode", tag="anode", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="cathode", tag="cathode", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region=reg, tag1="anode", tag2="cathode")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=dev)

    # --- material parameters
    SetSiliconParameters(dev, reg, T)
    ni = np.sqrt(2.86e19 * 2.66e19) * (T / 300) ** 1.5 * np.exp(-1.12 * Q / (2 * KB * T))
    for name, value in [("Permittivity", 11.8 * EPS0), ("ElectronCharge", Q), ("n_i", ni),
                        ("n1", ni), ("p1", ni), ("taun", tau), ("taup", tau),
                        ("T", T), ("V_t", KB * T / Q), ("kT", KB * T)]:
        devsim.set_parameter(device=dev, region=reg, name=name, value=value)

    # --- doping and mobility (node values, averaged on the mesh edges)
    CreateNodeModel(dev, reg, "Acceptors", f"{Na}*step({LENGTH / 2}-x)")
    CreateNodeModel(dev, reg, "Donors", f"{Nd}*step(x-{LENGTH / 2})")
    CreateNodeModel(dev, reg, "NetDoping", "Donors-Acceptors")
    CreateNodeModel(dev, reg, "Ntot", "Donors+Acceptors")
    CreateNodeModel(dev, reg, "mu_n_node", mobility(1430, 4.61e17, 1.52e15, T))
    CreateNodeModel(dev, reg, "mu_p_node", mobility(495, 1e17, 6.25e14, T))
    for carrier in ("n", "p"):
        devsim.edge_from_node_model(device=dev, region=reg, node_model=f"mu_{carrier}_node")
        CreateEdgeModel(dev, reg, f"mu_{carrier}_edge", f"0.5*(mu_{carrier}_node@n0+mu_{carrier}_node@n1)")

    def solve():
        devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=40)

    # --- step 1: equilibrium, Poisson equation only
    CreateSolution(dev, reg, "Potential")
    CreateSiliconPotentialOnly(dev, reg)
    for c in ("anode", "cathode"):
        CreateSiliconPotentialOnlyContact(dev, reg, c)
        devsim.set_parameter(device=dev, name=GetContactBiasName(c), value=0.0)
    devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=40)

    # --- step 2: full drift-diffusion model, carriers initialised from equilibrium
    CreateSolution(dev, reg, "Electrons")
    CreateSolution(dev, reg, "Holes")
    devsim.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    devsim.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(dev, reg, "mu_n_edge", "mu_p_edge")
    for c in ("anode", "cathode"):
        CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    solve()

    # --- step 3: ramp the bias from 0 V, separately into forward and reverse direction
    J = {}
    for branch in (sorted(v for v in voltages if v >= 0), sorted((v for v in voltages if v < 0), reverse=True)):
        v_now = 0.0
        devsim.set_parameter(device=dev, name=GetContactBiasName("anode"), value=0.0)
        solve()
        for v in branch:
            step = v - v_now
            while abs(v - v_now) > 1e-12:
                v_try = v_now + step if abs(step) < abs(v - v_now) else v
                devsim.set_parameter(device=dev, name=GetContactBiasName("anode"), value=v_try)
                try:
                    solve()
                    v_now = v_try
                except devsim.error:
                    step /= 2  # Newton did not converge: halve the bias step
                    if abs(step) < 1e-4:
                        raise
            J[v] = sum(devsim.get_contact_current(device=dev, contact="anode", equation=eq)
                       for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"))
    devsim.delete_device(device=dev)
    devsim.delete_mesh(mesh=mesh)
    return np.array([J[v] for v in voltages])


# --- helpers used in the rest of the notebook
VOLTAGES = np.sort(comsol.V_V.unique())  # the 23 voltages of the COMSOL dataset
J_MIN = 1e-7  # A/cm²: smaller currents are below the numerical resolution of the simulators


def devsim_like(df_device):
    """Simulates the diode of one COMSOL device with DEVSIM; returns a DataFrame in the same format."""
    p = df_device.iloc[0]
    J = simulate_diode(p.Na_cm3, p.Nd_cm3, p.tau_s, p.T_K, VOLTAGES)
    out = pd.DataFrame({"V_V": VOLTAGES, "J_A_cm2": J})
    for col in ["device_id", "Na_cm3", "Nd_cm3", "tau_s", "T_K"]:
        out[col] = p[col]
    return out


def curve(Na, Nd, tau, T, V):
    """Input table of one diode (or a parameter sweep) for the neural network."""
    return pd.DataFrame({"Na_cm3": Na, "Nd_cm3": Nd, "tau_s": tau, "T_K": T, "V_V": V})


# %% [markdown]
# ### Does DEVSIM agree with COMSOL?
# We simulate the first COMSOL diode with exactly the same parameters and voltages.

# %%
g = comsol[comsol.device_id == "d0001"].sort_values("V_V")
t0 = time.time()
d = devsim_like(g)
print(f"DEVSIM simulation of one I–V curve: {time.time() - t0:.1f} s")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
plot_iv(ax1, g, "COMSOL", "o", color="tab:blue")
plot_iv(ax1, d, "DEVSIM", "-", color="tab:red")
ax1.legend()
ax1.set_title("Diode d0001")
ax2.plot(g.V_V, d.J_A_cm2.values / g.J_A_cm2.values, "o-")
ax2.set_xlabel("V (V)")
ax2.set_ylabel("J_DEVSIM / J_COMSOL")
ax2.set_title("Ratio of the two simulators")
fig.tight_layout()

# %% [markdown]
# Now the same for 10 randomly chosen COMSOL diodes. We look at the ratio $J_\mathrm{DEVSIM}/J_\mathrm{COMSOL}$
# separately in forward bias ($V>0$) and in reverse bias ($V<0$), using only currents above the resolution
# limit, $|J|\ge J_\mathrm{min}=10^{-7}\ \mathrm{A/cm^2}$.

# %%
check_ids = rng.choice(comsol.device_id.unique(), 10, replace=False)
rows = []
for dev in check_ids:
    g = comsol[comsol.device_id == dev].sort_values("V_V")
    J_c, V_c = g.J_A_cm2.values, g.V_V.values
    ratio = devsim_like(g).J_A_cm2.values / J_c
    ok = np.abs(J_c) >= J_MIN
    fwd, rev = ok & (V_c > 0), ok & (V_c < 0)
    rows.append({"device_id": dev,
                 "forward: max |ratio-1|": np.max(np.abs(ratio[fwd] - 1)) if fwd.any() else np.nan,
                 "reverse: max |ratio-1|": np.max(np.abs(ratio[rev] - 1)) if rev.any() else np.nan,
                 "reverse points resolved": f"{rev.sum()}/{(V_c < 0).sum()}"})
pd.DataFrame(rows).round(4)

# %% [markdown]
# Two independent simulators solving the same model agree very well wherever the current is above the
# resolution limit (`NaN` = no reverse-bias point of that diode is above $J_\mathrm{min}$). Reverse currents
# are large enough only for short lifetimes or high temperatures – as the generation current
# $J_{\mathrm{gen}}\approx q n_i W/(2\tau)$ predicts.

# %% [markdown]
# ## 6 Your own DEVSIM dataset
#
# We sample `N_DIODES` new diodes by Latin hypercube sampling in the same parameter ranges and simulate them.
# This takes a few minutes (about 2–3 s per diode).

# %%
N_DIODES = 80
sampler = qmc.LatinHypercube(d=4, seed=7)
u = sampler.random(N_DIODES)
params = pd.DataFrame({
    "Na_cm3": 10 ** (15 + 3 * u[:, 0]),
    "Nd_cm3": 10 ** (15 + 3 * u[:, 1]),
    "tau_s": 10 ** (-8 + 3 * u[:, 2]),
    "T_K": 273.15 + 100 * u[:, 3],
})
params["device_id"] = [f"s{i + 1:04d}" for i in range(N_DIODES)]

t0 = time.time()
curves = []
for i, p in params.iterrows():
    try:
        J = simulate_diode(p.Na_cm3, p.Nd_cm3, p.tau_s, p.T_K, VOLTAGES)
    except Exception as err:  # keep going if a diode does not converge
        print(f"{p.device_id} failed: {err}")
        continue
    curves.append(pd.DataFrame({"device_id": p.device_id, "Na_cm3": p.Na_cm3, "Nd_cm3": p.Nd_cm3,
                                "tau_s": p.tau_s, "T_K": p.T_K, "V_V": VOLTAGES, "J_A_cm2": J}))
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{N_DIODES} diodes, {time.time() - t0:.0f} s")
devsim_data = pd.concat(curves, ignore_index=True)
devsim_data.to_csv("devsim_diodes.csv", index=False)
print(f"{devsim_data.device_id.nunique()} diodes simulated in {time.time() - t0:.0f} s, saved to devsim_diodes.csv")

# %% [markdown]
# ## 7 Training a neural-network surrogate
#
# ### Inputs and output
# * **Inputs** $x$: $\log_{10}N_A$, $\log_{10}N_D$, $\log_{10}\tau$, $T/300\,\mathrm{K}$, $V$ and $qV/k_BT$.
#   Doping and lifetime span orders of magnitude, so we use their logarithms. The last input carries no new
#   information – it is computed from $V$ and $T$ – but the diode current depends on exactly this combination,
#   $J\propto\exp(qV/n_{\mathrm{id}}k_BT)$. Giving it to the network directly (*physics-informed feature
#   engineering*) makes learning easier. All inputs are standardized to zero mean and unit variance.
# * **Output** $y$: the current density spans >10 orders of magnitude and changes sign. Training directly on
#   $J$ would make the network ignore all small currents. We use the transformation
#   $$y = \operatorname{asinh}\!\left(\frac{J}{J_s}\right),\qquad J_s = 10^{-8}\ \mathrm{A/cm^2},$$
#   which behaves like $\ln(2J/J_s)$ for large positive $J$, like $-\ln(2|J|/J_s)$ for large negative $J$, and is
#   smooth around zero. The scale $J_s$ is chosen near the numerical resolution limit: currents much smaller
#   than $J_s$ are mapped close to $y=0$, so the network does not waste effort on fitting numerical noise.
# * **Error measure**: the relative error of $J$, evaluated only where $|J|\ge J_\mathrm{min}=10^{-7}\ \mathrm{A/cm^2}$.
#
# ### Train / validation / test split
# The split is done **by diode**, not by row: the test diodes are never seen during training, so the test
# error tells how well the network generalizes to *new designs*.
#
# ### Network
# A fully connected network with 4 hidden layers of 128 neurons and the SiLU activation, trained with the Adam
# optimizer on the mean squared error of $y$ (full-batch gradient steps, learning rate decreasing along a
# cosine curve). We keep the weights with the lowest validation error. Training takes a few minutes on a CPU.

# %%
J_SCALE = 1e-8  # A/cm², scale of the asinh target transformation


def make_inputs(df):
    V_thermal = KB * df.T_K / Q  # thermal voltage kT/q (V)
    return np.column_stack([np.log10(df.Na_cm3), np.log10(df.Nd_cm3), np.log10(df.tau_s),
                            df.T_K / 300.0, df.V_V, df.V_V / V_thermal]).astype(np.float32)


def to_target(J):
    return np.arcsinh(J / J_SCALE).astype(np.float32)


def from_target(y):
    return np.sinh(y) * J_SCALE


def split_by_device(df, fractions=(0.7, 0.15, 0.15), seed=0):
    ids = np.random.default_rng(seed).permutation(df.device_id.unique())
    n1 = int(fractions[0] * len(ids))
    n2 = n1 + int(fractions[1] * len(ids))
    parts = ids[:n1], ids[n1:n2], ids[n2:]
    return [df[df.device_id.isin(p)].reset_index(drop=True) for p in parts]


class Surrogate(torch.nn.Module):
    def __init__(self, x_train, width=128, depth=4):
        super().__init__()
        self.register_buffer("x_mean", torch.tensor(x_train.mean(0)))
        self.register_buffer("x_std", torch.tensor(x_train.std(0)))
        layers, n_in = [], x_train.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.SiLU()]
            n_in = width
        layers.append(torch.nn.Linear(n_in, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std).squeeze(-1)

    def predict_J(self, df):
        with torch.no_grad():
            y = self(torch.tensor(make_inputs(df))).numpy()
        return from_target(y)


def train_surrogate(train_df, val_df, epochs=4000, lr=3e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    x_tr, y_tr = torch.tensor(make_inputs(train_df)), torch.tensor(to_target(train_df.J_A_cm2.values))
    x_va, y_va = torch.tensor(make_inputs(val_df)), torch.tensor(to_target(val_df.J_A_cm2.values))
    model = Surrogate(x_tr.numpy())
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
        if verbose and (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1}: train MSE {loss.item():.4f}, validation MSE {val:.4f}")
    model.load_state_dict(best_state)
    if verbose:
        print(f"training time {time.time() - t0:.0f} s, best validation MSE {best:.4f}")
    return model, np.array(history)


def errors(model, df):
    """Relative error of the predicted current for |J| >= J_MIN, separately in forward and reverse bias."""
    ok = np.abs(df.J_A_cm2.values) >= J_MIN
    J_true, J_pred, V = df.J_A_cm2.values[ok], model.predict_J(df[ok]), df.V_V.values[ok]
    rel = np.abs(J_pred / J_true - 1)
    fwd, rev = V > 0, V < 0
    return {"forward: median rel. error": np.median(rel[fwd]),
            "forward: 95th percentile": np.percentile(rel[fwd], 95),
            "reverse: median rel. error": np.median(rel[rev]),
            "reverse: 95th percentile": np.percentile(rel[rev], 95)}


# %% [markdown]
# ### Model A: trained on the COMSOL data

# %%
c_train, c_val, c_test = split_by_device(comsol)
print(f"COMSOL diodes: {c_train.device_id.nunique()} train, {c_val.device_id.nunique()} validation, "
      f"{c_test.device_id.nunique()} test")
model_A, hist_A = train_surrogate(c_train, c_val)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(hist_A[:, 0], label="training")
ax.semilogy(hist_A[:, 1], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE of asinh(J/J_s)")
ax.legend()
ax.set_title("Training history, model A")

# %% [markdown]
# ### How good is model A on the test diodes?

# %%
pd.Series(errors(model_A, c_test)).to_frame("model A on COMSOL test diodes").style.format("{:.2%}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
J_pred = model_A.predict_J(c_test)
sc = ax1.scatter(np.abs(c_test.J_A_cm2), np.abs(J_pred), c=c_test.V_V, s=6, cmap="coolwarm")
lims = [1e-12, 1e4]
ax1.plot(lims, lims, "k--", lw=1)
ax1.set(xscale="log", yscale="log", xlim=lims, ylim=lims, xlabel="|J| COMSOL (A/cm²)",
        ylabel="|J| DNN (A/cm²)", title="Test diodes: prediction vs. simulation")
fig.colorbar(sc, ax=ax1, label="V (V)")

v_fine = np.linspace(-2, 0.8, 200)
for dev, color in zip(c_test.device_id.unique()[:3], ["tab:blue", "tab:green", "tab:red"]):
    g = c_test[c_test.device_id == dev]
    plot_iv(ax2, g, f"{dev} COMSOL", "o", color=color)
    fine = pd.DataFrame({c: g.iloc[0][c] for c in ["Na_cm3", "Nd_cm3", "tau_s", "T_K"]}, index=range(len(v_fine)))
    fine["V_V"] = v_fine
    ax2.semilogy(v_fine, np.abs(model_A.predict_J(fine)), "-", color=color, label=f"{dev} DNN")
ax2.legend(fontsize=7)
ax2.set_title("Three test diodes: simulation points and DNN curves")
fig.tight_layout()

# %% [markdown]
# The network also predicts the current **between** the simulated voltages (solid lines) – it has learned a
# smooth function of $V$, not just the 23 training voltages. Near $V=0$ the current crosses zero and is below
# the resolution limit; the relative error there is not meaningful, which is why the error table uses only
# $|J|\ge J_\mathrm{min}$.

# %% [markdown]
# ## 8 COMSOL data versus your DEVSIM data
#
# Model B has the same architecture, but is trained on your DEVSIM diodes. We evaluate both models on test
# diodes from **both** simulators.

# %%
s_train, s_val, s_test = split_by_device(devsim_data)
model_B, hist_B = train_surrogate(s_train, s_val)

table = pd.DataFrame({
    "A (trained on COMSOL) → COMSOL test": errors(model_A, c_test),
    "A (trained on COMSOL) → DEVSIM test": errors(model_A, s_test),
    "B (trained on DEVSIM) → COMSOL test": errors(model_B, c_test),
    "B (trained on DEVSIM) → DEVSIM test": errors(model_B, s_test),
}).T
table.style.format("{:.2%}")

# %% [markdown]
# **Things to notice:**
# * Model A was trained on many more diodes (350) than model B (~56). How much does the amount of training
#   data matter?
# * Both simulators solve the same physical model, so a network trained on one should work on the other.
#   If the errors across simulators are similar to the errors within one simulator, the difference between
#   the simulators is smaller than the error of the network itself.

# %% [markdown]
# ## 9 Speed: simulator versus surrogate

# %%
n_curves = 1000
batch = pd.DataFrame({
    "Na_cm3": np.repeat(10 ** rng.uniform(15, 18, n_curves), len(VOLTAGES)),
    "Nd_cm3": np.repeat(10 ** rng.uniform(15, 18, n_curves), len(VOLTAGES)),
    "tau_s": np.repeat(10 ** rng.uniform(-8, -5, n_curves), len(VOLTAGES)),
    "T_K": np.repeat(rng.uniform(273, 373, n_curves), len(VOLTAGES)),
    "V_V": np.tile(VOLTAGES, n_curves)})
t0 = time.time()
model_A.predict_J(batch)
t_dnn = (time.time() - t0) / n_curves
t0 = time.time()
simulate_diode(1e16, 1e16, 1e-6, 300.0, VOLTAGES)
t_sim = time.time() - t0
print(f"DEVSIM: {t_sim * 1e3:.0f} ms per I–V curve, DNN: {t_dnn * 1e6:.1f} µs per I–V curve "
      f"→ about {t_sim / t_dnn:,.0f}× faster")

# %% [markdown]
# ## 10 Did the network learn physics?
#
# A small prediction error does not guarantee physically sensible behaviour. We test two predictions of
# Section 1 on a diode with $N_A=N_D=10^{16}\ \mathrm{cm^{-3}}$ and $\tau=1\ \mu\mathrm{s}$.
#
# **(a) Ideality factor.** From $J\propto\exp(qV/n_{\mathrm{id}}k_BT)$:
# $\;n_{\mathrm{id}}(V) = \dfrac{q}{k_BT}\left(\dfrac{d\ln J}{dV}\right)^{-1}$.
# We expect values between 1 and 2 in moderate forward bias, and a rise at high current.
#
# **(b) Temperature dependence.** At fixed forward voltage the current rises steeply with temperature,
# because $J_0\propto n_i^2\propto T^3\exp(-E_g/k_BT)$.

# %%
V = np.linspace(0.1, 0.8, 71)
T = 300.0
J_dnn = model_A.predict_J(curve(1e16, 1e16, 1e-6, T, V))
J_sim = simulate_diode(1e16, 1e16, 1e-6, T, V)
n_id = lambda J: Q / (KB * T) / np.gradient(np.log(J), V)

temps = np.array([275.0, 300.0, 325.0, 350.0, 370.0])
J_T_sim = np.array([simulate_diode(1e16, 1e16, 1e-6, t, [0.4])[0] for t in temps])
t_fine = np.linspace(273, 373, 50)
J_T_dnn = model_A.predict_J(curve(1e16, 1e16, 1e-6, t_fine, 0.4))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(V, n_id(J_sim), "k-", label="DEVSIM")
ax1.plot(V, n_id(J_dnn), "r--", label="DNN (model A)")
ax1.set(xlabel="V (V)", ylabel="ideality factor", ylim=(0.5, 4), title="(a) Ideality factor at 300 K")
ax1.legend()
ax2.semilogy(temps, J_T_sim, "ko", label="DEVSIM")
ax2.semilogy(t_fine, J_T_dnn, "r-", label="DNN (model A)")
ax2.set(xlabel="T (K)", ylabel="J at V = 0.4 V (A/cm²)", title="(b) Temperature dependence")
ax2.legend()
fig.tight_layout()

# %% [markdown]
# The ideality factor is a *derivative* of the learned function. Derivatives are much more sensitive to small
# errors of the network than the function values themselves – compare the two curves in (a).

# %% [markdown]
# ## 11 Summary and questions
#
# * A device simulator solves the drift–diffusion equations with finite volumes, Newton's method and bias
#   ramping. Two independent simulators (COMSOL and DEVSIM) solving the same model agree closely.
# * A small neural network trained on a few hundred simulated diodes predicts their I–V curves with an error
#   of a few percent and is many orders of magnitude faster than the simulator.
# * The choice of the target transformation (asinh), the split by device and the check of physical behaviour
#   are as important as the network itself.
#
# **Questions**
# 1. Why does the reverse current depend strongly on the lifetime $\tau$, while the forward current at
#    0.6 V depends on it much less? (Hint: Section 1 and the plot in Section 4.)
# 2. Which input parameter range would you extend if you wanted to use the surrogate for power diodes?
#    What would you have to check in the simulator first?
# 3. Model B was trained on fewer diodes. Increase `N_DIODES` in Section 6, rerun Sections 6–8 and describe
#    how the errors change.
# 4. What happens to the prediction for a temperature outside the training range, e.g. 450 K? Try it with
#    `model_A.predict_J(curve(1e16, 1e16, 1e-6, 450.0, V))` and explain the result.
