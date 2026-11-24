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
# # Magnetic materials – from the electron spin to transformers and memory chips
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zemenja1/ppsr-ai-svs/blob/main/notebooks/04_magnetism.ipynb)
#
# *Teaching materials of the project “AI ve výuce – Elektrotechnologie”, CTU in Prague, FEE.*
# *Draft version (November 2026). License: text CC BY 4.0, code MIT.*
#
# This notebook is the **study material of the two lectures on magnetic materials**. It contains the theory –
# definitions, equations, typical material values – and draws every diagram **from the physics**, from a few lines
# of NumPy up to micromagnetic simulations, COMSOL device models and neural-network surrogates. Change a parameter
# and see what happens.
#
# | Section | Content | Method |
# |---|---|---|
# | **Part I** | **Where magnetism comes from (lecture 1)** | |
# | 1 B, H and M | dipole and current loop, Ampère's currents | closed-form fields |
# | 2 Demagnetizing field, units | demagnetizing factor of an ellipsoid, SI and cgs | formulas |
# | 3 Stray field of a magnetized prism | surface charges, $H_d = -NM$ | closed-form magnetostatics |
# | 4 Susceptibility and permeability | χ(T), table of µr | formulas |
# | 5 Atomic origin, magnetic order | Bohr magneton, types of magnetism, liquid oxygen (video) | numbers |
# | 6 Exchange and the Curie temperature | Heisenberg model, mean-field M(T) | mean-field model |
# | **Part II** | **From the energy budget to devices and data (lecture 2)** | |
# | 7 Energy terms, length scales | exchange, anisotropy, Zeeman, stray field | numbers |
# | 8 Shape anisotropy, Stoner–Wohlfarth | square loop, astroid | Stoner–Wohlfarth model |
# | 9 Crystal anisotropy, magnetostriction | anisotropy energy surfaces of Fe, Co, Ni | formulas |
# | 10 Shape vs crystal anisotropy | $K_{eff} = K_{sh} - K_u$ | Stoner–Wohlfarth model |
# | 11 Exchange: micromagnetics | single-domain size, vortex, two domains | magnum.np |
# | 12 Domains | why domains form, domain width | Kittel model |
# | 13 The domain wall | wall width $\pi\sqrt{A/K}$ | magnum.np, COMSOL |
# | 14 Wall motion | LLG equation, wall velocity | magnum.np, formula |
# | 15 Rotation vs wall propagation | two coercive fields, pinning | magnum.np, COMSOL |
# | 16 The hysteresis loop | Bs, Br, Hc, µi, µmax, soft vs hard | model loop |
# | 17 From the loop to the loss | coercivity ladder, hysteresis loss | numbers |
# | 18 Core losses | hysteresis, eddy-current and excess loss, skin depth | Bertotti model |
# | 19 Transformer and generator | COMSOL vs a second method, DNN surrogates | COMSOL, magnetic circuit, own FEM, PyTorch |
# | 20 Permanent magnets | second quadrant, (BH)max, load lines | formulas |
# | 21 Materials in numbers | soft and hard materials, history of (BH)max | data |
# | 22 Recording and spintronics | superparamagnetic limit, GMR/TMR, STT-MRAM | Néel–Arrhenius, macrospin LLGS, magnum.np |
#
# **How to work with this notebook:** run the cells from top to bottom (`Shift+Enter`). Many sections start with a
# **prediction question** – write down your answer *before* you run the cells, then compare. Part I and
# sections 7–10 need only NumPy/SciPy and run in seconds. From section 11 on we use the micromagnetic code
# [magnum.np](https://gitlab.com/magnum.np/magnum.np) (installed with `pip`); its live simulations take one to a few
# minutes each on the free Colab CPU. The larger simulations were done in advance and their results are loaded from
# the repository, together with reference results of COMSOL Multiphysics. The whole notebook runs in about
# 20–25 minutes on a CPU, most of it in sections 11–15 and 19.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

MU0 = 4e-7 * np.pi                   # vacuum permeability (T m/A)
MU_B = 9.2740100783e-24              # Bohr magneton (J/T)
K_B = 1.380649e-23                   # Boltzmann constant (J/K)
E, M_E, HBAR = 1.602176634e-19, 9.1093837015e-31, 1.054571817e-34
BLUE, ORANGE, GREY = "#1f4e8c", "#e07b10", "#7f7f7f"
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

# Iron (room temperature), the values of the COMSOL ribbon model used in sections 13-15. The COMSOL micromagnetics
# plugin takes A and K in field units: A = 1.96e-11 A*m and H_K = 2K/(mu0 Ms) = 43.7 kA/m.
Ms = 1.71e6                       # saturation magnetization (A/m)
A_ex = 0.5 * MU0 * Ms * 1.96e-11  # exchange stiffness (J/m), = 2.1e-11
K1 = 0.5 * MU0 * Ms * 4.37e4      # anisotropy constant (J/m^3), = 4.7e4; in Fe it is cubic, here uniaxial

# %% [markdown]
# # Part I – Where magnetism comes from (lecture 1)

# %% [markdown]
# ## 1 B, H and M
#
# The **magnetic moment** $\mathbf m$ (A·m²) is the elementary magnetic quantity; the **magnetization**
# $\mathbf M$ (A/m) is the moment per volume, averaged over a small region. The magnetic flux density
# $\mathbf B$ (T) and the field $\mathbf H$ (A/m) are related by
# $$\mathbf B = \mu_0(\mathbf H + \mathbf M),\qquad \nabla\cdot\mathbf B = 0,\qquad
#   \nabla\times\mathbf H = \mathbf j_c\quad(\text{magnetostatics}).$$
# Why two fields? The sources of $\mathbf B$ are *all* currents, $\nabla\times\mathbf B = \mu_0(\mathbf j_c +
# \mathbf j_m)$: the conduction currents $\mathbf j_c$ that we can measure, and the magnetization currents
# $\mathbf j_m = \nabla\times\mathbf M$ inside the material, which we cannot (they are mostly the spin of the
# electrons). Defining $\mathbf H = \mathbf B/\mu_0 - \mathbf M$ removes $\mathbf j_m$ from Ampère's law:
# $\oint\mathbf H\cdot\mathrm d\mathbf l = I_c$.
#
# A small current loop (current $I$, area $\mathcal A$) is a magnetic dipole with $m = I\mathcal A$; far away
# (distance ≫ loop size) both have the dipole field
# $$\mathbf B(\mathbf r) = \frac{\mu_0}{4\pi}\left[\frac{3(\mathbf m\cdot\mathbf r)\,\mathbf r}{r^5}
#   - \frac{\mathbf m}{r^3}\right].$$
# Below: the dipole field and the exact field of a circular loop with the same moment (elliptic integrals). The
# lines of $\mathbf B$ are always closed – there are no magnetic charges.

# %%
def dipole_B(x, z, m=1.0):
    """B (T) of a point dipole m (A m^2) along z at the origin, in the x-z plane."""
    r = np.hypot(x, z) + 1e-30
    pre = MU0 * m / (4 * np.pi * r**5)
    return pre * 3 * x * z, pre * (3 * z * z - r * r)


def loop_B(x, z, a=1.0, I=1.0):
    """Exact B (T) of a circular loop (radius a, current I) in the plane z = 0, at radius |x| and height z."""
    rho = np.abs(x) + 1e-12
    k2 = 4 * a * rho / ((a + rho)**2 + z**2)
    K, Ek = ellipk(k2), ellipe(k2)
    den = np.sqrt((a + rho)**2 + z**2)
    c = MU0 * I / (2 * np.pi * den)
    Bz = c * (K + (a * a - rho * rho - z * z) / ((a - rho)**2 + z**2) * Ek)
    Br = c * z / rho * (-K + (a * a + rho * rho + z * z) / ((a - rho)**2 + z**2) * Ek)
    return np.sign(x) * Br, Bz


g = np.linspace(-3, 3, 300)
X, Z = np.meshgrid(g, g)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (Bx, Bz), title in [(ax[0], dipole_B(X, Z, np.pi), "point dipole, m = π A·m²"),
                           (ax[1], loop_B(X, Z), "current loop, a = 1 m, I = 1 A (m = π A·m²)")]:
    a.streamplot(g, g, Bx, Bz, color=BLUE, density=1.4, linewidth=0.8)
    a.set_aspect("equal"); a.set_title(title); a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
ax[0].arrow(0, -0.3, 0, 0.6, width=0.05, color=ORANGE, zorder=5)
ax[1].plot([-1, 1], [0, 0], "o", color=ORANGE, ms=8)
zz = np.linspace(0.05, 5, 200)
ax[2].semilogy(zz, dipole_B(0 * zz, zz, np.pi)[1] * 1e9, color=BLUE, label="dipole")
ax[2].semilogy(zz, loop_B(0 * zz + 1e-9, zz)[1] * 1e9, "--", color=ORANGE, label="loop (exact)")
ax[2].set_xlabel("z on the axis (m)"); ax[2].set_ylabel("B_z (nT)"); ax[2].legend()
ax[2].set_title("on the axis: 28 % apart at z = 2a, 6 % at z = 5a")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Are there currents inside a magnet?** A uniformly magnetized bar is equivalent to a surface current $M$ per
# metre of length (like a solenoid). For iron, $M_s = 1.71\times10^{6}$ A/m:

# %%
Ms_Fe = 1.71e6
print(f"equivalent surface current of a 1 cm long iron bar: {Ms_Fe * 0.01:.1e} A")
print("a solenoid of 100 turns on 1 cm would need", f"{Ms_Fe * 0.01 / 100:.0f} A per turn for the same B")

# %% [markdown]
# No real conductor carries 17 kA around a 1 cm bar without melting – the “currents” of a magnet are the spins
# of its electrons (section 5).

# %% [markdown]
# ## 2 Demagnetizing field and units
#
# Inside a magnet, $\mathbf H = \mathbf H' + \mathbf H_d$: the applied field plus the **demagnetizing field**
# created by the magnetization itself. Only inside an ellipsoid is $\mathbf H_d$ uniform,
# $H_{d,i} = -N_{ij}M_j$, with the demagnetizing tensor $N$ (eigenvalues $N_x + N_y + N_z = 1$). For a spheroid
# with the symmetry axis $z$ and the aspect ratio $r = c/a$ the factor $N_z$ has a closed form; the other two are
# $N_x = N_y = (1 - N_z)/2$. (For a prism – as in section 3 – the field is not uniform, and $N$ is
# its volume average.)

# %%
def Nz_spheroid(r):
    """Demagnetizing factor along the symmetry axis of a spheroid with aspect ratio r = c/a."""
    r = np.asarray(r, float)
    out = np.full_like(r, 1 / 3)
    p, o = r > 1 + 1e-9, r < 1 - 1e-9
    e = np.sqrt(r[p]**2 - 1)
    out[p] = (r[p] / e * np.log(r[p] + e) - 1) / (r[p]**2 - 1)
    e = np.sqrt(1 - r[o]**2)
    out[o] = (1 - r[o] / e * np.arccos(r[o])) / (1 - r[o]**2)
    return out


r = np.logspace(-2, 2, 400)
plt.figure(figsize=(7, 4))
plt.semilogx(r, Nz_spheroid(r), color=BLUE, label="N_z (along the axis)")
plt.semilogx(r, (1 - Nz_spheroid(r)) / 2, color=ORANGE, label="N_x = N_y")
for rr, lab in [(0.01, "thin film\nN_z → 1"), (1, "sphere\n1/3"), (100, "needle\nN_z → 0")]:
    plt.annotate(lab, (rr, Nz_spheroid(np.array([rr]))[0]), textcoords="offset points", xytext=(5, 8), fontsize=8)
plt.xlabel("aspect ratio c/a of the spheroid"); plt.ylabel("demagnetizing factor"); plt.legend(); plt.show()

# %% [markdown]
# | Shape | magnetized | $N$ |
# |---|---|---|
# | long needle | along the axis / perpendicular | 0 / 1/2 |
# | sphere | any direction | 1/3 |
# | thin film | in plane / perpendicular | 0 / 1 |
#
# The demagnetizing energy $\tfrac12\mu_0 N M^2$ per volume is smallest along the direction with the smallest $N$:
# an elongated magnet prefers to be magnetized along its long axis (*shape anisotropy*, section 8).
#
# **Units.** Older literature and many data sheets use Gaussian (cgs) units:
#
# | Quantity | SI unit | cgs unit | 1 cgs unit = |
# |---|---|---|---|
# | flux density $B$ | T | G (gauss) | $10^{-4}$ T |
# | field $H$ | A/m | Oe (oersted) | $10^3/4\pi \approx 79.6$ A/m |
# | magnetization $M$ | A/m | emu/cm³ | $10^3$ A/m |
# | polarization $J = \mu_0M$ | T | – | – |
# | susceptibility $\chi$ (volume) | 1 | 1 | $\chi_{SI} = 4\pi\chi_{cgs}$ |
# | energy product $(BH)_{max}$ | kJ/m³ | MGOe | $100/4\pi \approx 7.96$ kJ/m³ |

# %% [markdown]
# ## 3 The stray field of a uniformly magnetized prism
#
# **Predict:** inside a bar magnet magnetized along its axis, do $\mathbf H$ and $\mathbf B$ point in the same
# direction?
#
# A uniform magnetization $\mathbf M$ acts like **magnetic surface charges** $\sigma = \mathbf M\cdot\mathbf n$ on
# the faces of the body. The field of a uniformly charged rectangle has a closed form, so the field of a prism is
# a sum over its faces. Inside an ellipsoid the field is uniform, $\mathbf H_d = -N\mathbf M$; for a prism it is
# almost uniform and the volume average defines the **demagnetizing factors** $N_x, N_y, N_z$
# (with $N_x + N_y + N_z = 1$), known analytically (Aharoni 1998).

# %%
def demag_factor_z(a, b, c):
    """Aharoni (1998): demagnetizing factor along z of a prism with edges a, b, c (along x, y, z)."""
    a, b, c = a / 2, b / 2, c / 2
    r, ab, bc, ac = np.sqrt(a*a + b*b + c*c), np.sqrt(a*a + b*b), np.sqrt(b*b + c*c), np.sqrt(a*a + c*c)
    D = ((b*b - c*c) / (2*b*c) * np.log((r - a) / (r + a)) + (a*a - c*c) / (2*a*c) * np.log((r - b) / (r + b))
         + b / (2*c) * np.log((ab + a) / (ab - a)) + a / (2*c) * np.log((ab + b) / (ab - b))
         + c / (2*a) * np.log((bc - b) / (bc + b)) + c / (2*b) * np.log((ac - a) / (ac + a))
         + 2 * np.arctan(a*b / (c*r)) + (a**3 + b**3 - 2*c**3) / (3*a*b*c) + (a*a + b*b - 2*c*c) / (3*a*b*c) * r
         + c / (a*b) * (ac + bc) - (ab**3 + bc**3 + ac**3) / (3*a*b*c))
    return D / np.pi


def demag_factors(a, b, c):
    """(Nx, Ny, Nz) of a prism with edges a, b, c."""
    return np.array([demag_factor_z(b, c, a), demag_factor_z(c, a, b), demag_factor_z(a, b, c)])


def _rect_field(u1, u2, v1, v2, w, sigma):
    """Field of a rectangle with surface charge sigma (A/m) in the plane w = 0; u1 = u - u_min, u2 = u - u_max..."""
    Hu, Hv, Hw = 0.0, 0.0, 0.0
    for u, su in ((u1, 1), (u2, -1)):
        for v, sv in ((v1, 1), (v2, -1)):
            R = np.sqrt(u*u + v*v + w*w) + 1e-30
            s = su * sv
            Hu = Hu - s * np.log(np.abs(v + R) + 1e-30)
            Hv = Hv - s * np.log(np.abs(u + R) + 1e-30)
            Hw = Hw + s * np.arctan(u * v / (w * R + np.where(w >= 0, 1e-30, -1e-30)))
    return sigma / (4 * np.pi) * np.stack(np.broadcast_arrays(Hu, Hv, Hw))


def prism_field(x, y, z, dims, M):
    """Field H (A/m) of a prism centred at the origin, edges dims = (a, b, c), uniform magnetization M (A/m)."""
    P, h = np.array(np.broadcast_arrays(x, y, z), float), np.array(dims) / 2
    H = np.zeros_like(P)
    for k in range(3):                                  # the faces normal to axis k carry the charge +-M_k
        if M[k] == 0:
            continue
        i, j = [q for q in range(3) if q != k]
        for sgn in (1, -1):
            f = _rect_field(P[i] + h[i], P[i] - h[i], P[j] + h[j], P[j] - h[j], P[k] - sgn * h[k], sgn * M[k])
            H[i] += f[0]; H[j] += f[1]; H[k] += f[2]
    return H


dims = (5e-3, 5e-3, 20e-3)                              # the iron prism, long axis z
N = demag_factors(*dims)
print("demagnetizing factors (Aharoni):  Nx = %.4f  Ny = %.4f  Nz = %.4f   sum = %.4f" % (*N, N.sum()))

# numerical check: average of the field over the volume of the prism
n = 24
g = [((np.arange(n) + 0.5) / n - 0.5) * d for d in dims]
X, Y, Z = np.meshgrid(*g, indexing="ij")
for k, ax in enumerate("xyz"):
    M = [0, 0, 0]; M[k] = Ms
    print(f"M along {ax}: -<H_{ax}>/Ms = {-prism_field(X, Y, Z, dims, M)[k].mean() / Ms:.4f}")

# %% [markdown]
# The small difference comes from the coarse numerical average near the edges. Now let us look at the field
# itself, in the plane $y = 0$, for $\mathbf M$ along the long axis $z$.

# %%
# grid points between the faces of the prism, never on them: on a charged surface H jumps by M and the
# point would be "half inside" (H taken from one side, M from the other gives a spurious spike of B)
x = np.linspace(-12e-3, 12e-3, 121); z = np.arange(-21.9e-3, 22e-3, 0.2e-3)
XX, ZZ = np.meshgrid(x, z, indexing="ij")
H = prism_field(XX, 0 * XX, ZZ, dims, [0, 0, Ms])
inside = (np.abs(XX) < dims[0] / 2) & (np.abs(ZZ) < dims[2] / 2)
Mz = np.where(inside, Ms, 0.0)
Bx, Bz = MU0 * H[0], MU0 * (H[2] + Mz)

fig, ax = plt.subplots(1, 3, figsize=(13, 5))
for a, (U, V, title) in zip(ax[:2], [(H[0], H[2], "H (stray / demagnetizing field)"), (Bx, Bz, "B = µ0 (H + M)")]):
    # rows of the arrays = x, columns = z; the plot has z horizontal and x vertical
    a.streamplot(z * 1e3, x * 1e3, V, U, color=np.log10(np.hypot(U, V) + 1e-9), cmap="Blues",
                 density=1.4, linewidth=0.8)
    a.add_patch(plt.Rectangle((-10, -2.5), 20, 5, fill=False, ec=ORANGE, lw=2))
    a.set_title(title); a.set_xlabel("z (mm)"); a.set_ylabel("x (mm)"); a.set_aspect("equal")
ax[2].plot(z * 1e3, H[2, 60, :] / Ms, color=BLUE, label="H_z / Ms")
ax[2].plot(z * 1e3, Bz[60, :] / (MU0 * Ms), color=ORANGE, label="B_z / (µ0 Ms)")
ax[2].axhline(-N[2], color=BLUE, ls=":", label="-Nz")
ax[2].set_xlabel("z (mm) on the axis"); ax[2].legend(); ax[2].set_title("Along the axis")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What we see**
# * Outside, $\mathbf H$ and $\mathbf B$ are the same field (times $\mu_0$): it runs from the north end (positive
#   charges) to the south end.
# * Inside, $\mathbf H$ points **against** $\mathbf M$ – it is the *demagnetizing* field. $\mathbf B$ inside points
#   along $\mathbf M$; its field lines are closed loops, while $\mathbf H$ lines start and end on the surface
#   charges.
# * $H_d$ is not uniform in a prism (strongest near the end faces), but its average is $-N_zM_s$.
#
# The stray field stores the energy $E_d = \tfrac12\mu_0 M_s^2 V\,(N_x m_x^2 + N_y m_y^2 + N_z m_z^2)$. It is
# smallest when $\mathbf M$ lies along the long axis ($N_z$ is the smallest factor): **the shape alone makes the
# long axis an easy axis** – the *shape anisotropy*. Its strength is
# $K_{sh} = \tfrac12\mu_0M_s^2(N_x - N_z)$.
#
# **Check:** a cube has $N_x = N_y = N_z = 1/3$ – no shape anisotropy; magnetostatics has **no length scale**, so the
# factors do not change if we scale the prism from millimetres to nanometres.

# %%
print("cube:", demag_factors(1, 1, 1))
print("5 x 5 x 20 nm prism:", demag_factors(5e-9, 5e-9, 20e-9), "(the same as for millimetres)")
K_sh = 0.5 * MU0 * Ms**2 * (N[0] - N[2])
print(f"shape anisotropy of the prism K_sh = {K_sh:.3g} J/m^3  (compare K1 of iron = {K1:.2g} J/m^3)")
print(f"anisotropy field of the shape H_K = (Nx - Nz) Ms = {(N[0] - N[2]) * Ms / 1e3:.0f} kA/m")

# %% [markdown]
# ## 4 Susceptibility and permeability
#
# In a linear isotropic material $\mathbf M = \chi\mathbf H$ and
# $$\mathbf B = \mu_0(1+\chi)\mathbf H = \mu_0\mu_r\mathbf H .$$
# For ferromagnets $M(H)$ is nonlinear and has hysteresis, so $\mu_r$ depends on the field and on the history – data
# sheets give the **initial** and the **maximum** permeability (section 16). The temperature dependence of $\chi$
# separates the types of magnetism:
# * diamagnet: $\chi < 0$, small, independent of $T$;
# * paramagnet: Curie law $\chi = C/T$;
# * ferromagnet above the Curie temperature: Curie–Weiss law $\chi = C/(T - T_C)$, diverging at $T_C$;
# * antiferromagnet: $\chi = C/(T+\theta)$ above the Néel temperature $T_N$, a cusp at $T_N$.

# %%
T = np.linspace(1, 600, 600)
C = 1.0
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(T, C / T, color=BLUE); ax[0].set_title("paramagnet: χ = C/T")
Tc = 200
ax[1].plot(T[T > Tc + 3], C / (T[T > Tc + 3] - Tc), color=BLUE); ax[1].axvspan(0, Tc, color=GREY, alpha=0.2)
ax[1].text(20, 0.085, "ordered\n(M ≠ 0 at H = 0)", fontsize=8)
ax[1].axvline(Tc, color=ORANGE, ls=":"); ax[1].set_title("ferromagnet: χ = C/(T − T_C) above T_C")
TN, th = 250, 150
chi_af = np.where(T > TN, C / (T + th), C / (TN + th) * (0.67 + 0.33 * (T / TN)**2))
ax[2].plot(T, chi_af, color=BLUE); ax[2].axvline(TN, color=ORANGE, ls=":")
ax[2].set_title("antiferromagnet: cusp at T_N")
for a in ax:
    a.set_xlabel("T (K)"); a.set_ylabel("χ (arb. units)"); a.set_ylim(0, None)
ax[1].set_ylim(0, 0.1)
plt.tight_layout(); plt.show()

# %% [markdown]
# (The curve below $T_N$ is schematic: it depends on the field direction relative to the spin axis.)
#
# **Typical relative permeabilities** (room temperature, low field; ferromagnets: maximum values, depending
# strongly on purity, processing and field):
#
# | Material | $\mu_r$ |
# |---|---|
# | copper, water (diamagnetic) | $1 - 10^{-5}$ |
# | air | $1 + 4\times10^{-7}$ |
# | aluminium (paramagnetic) | $1 + 2\times10^{-5}$ |
# | NdFeB magnet (recoil) | ≈ 1.05 |
# | nickel | 100–600 |
# | NiZn / MnZn ferrite | 10–1000 / 1000–15 000 |
# | electrical steel (Fe-Si) | 4000–40 000 |
# | pure iron (annealed) | 5000–200 000 |
# | permalloy (Ni-Fe 80 %) | 10⁴–10⁶ |
# | amorphous / nanocrystalline alloys | 10⁴–10⁶ |

# %% [markdown]
# ## 5 Atomic origin of magnetism and magnetic order
#
# An electron on an orbit with angular momentum $\boldsymbol\ell$ forms a current loop; its moment and the moment of
# its spin $\mathbf s$ are
# $$\mathbf m_\ell = -\frac{e}{2m_e}\boldsymbol\ell,\qquad \mathbf m_s = -\frac{e}{m_e}\mathbf s,\qquad
#   \mathbf m = -\frac{\mu_B}{\hbar}(\boldsymbol\ell + 2\mathbf s),\qquad \mu_B = \frac{e\hbar}{2m_e},$$
# and the energy in a field (Zeeman energy) is $\mathcal H_Z = -\mathbf m\cdot\mathbf B$.

# %%
print(f"Bohr magneton mu_B = e*hbar/(2 m_e) = {E * HBAR / (2 * M_E):.4e} J/T")
n_Fe = 7874 / (55.845e-3 / 6.02214076e23)
print(f"atoms of iron per m^3: {n_Fe:.3e};  Ms / (n mu_B) = {Ms_Fe / (n_Fe * MU_B):.2f} Bohr magnetons per atom")
print(f"Zeeman energy of one mu_B in 1 T: {MU_B / E * 1e3:.3f} meV;  kT at 300 K: {K_B * 300 / E * 1e3:.1f} meV")

# %% [markdown]
# Iron carries 2.2 µB per atom. The Zeeman energy of one moment in 1 T is **450 times smaller than the thermal energy**
# – an applied field alone can align only a small fraction of free moments. This is the **paramagnet**: the
# moments $\mu = g J\mu_B$ follow the Brillouin function $M/M_{sat} = B_J(x)$, $x = gJ\mu_B B/k_BT$
# (for $J = 1/2$: $\tanh x$).

# %%
def brillouin(J, x):
    """Brillouin function B_J(x); series (J+1)/(3J) x for small x."""
    x = np.asarray(x, float)
    a, b = (2 * J + 1) / (2 * J), 1 / (2 * J)
    xs = np.where(np.abs(x) < 1e-3, 1.0, x)
    return np.where(np.abs(x) < 1e-3, (J + 1) / (3 * J) * x, 1 / np.tanh(a * xs) * a - 1 / np.tanh(b * xs) * b)


Bf = np.linspace(0, 10, 300)
plt.figure(figsize=(6.5, 4))
for Tk, c in [(2, BLUE), (10, ORANGE), (300, "k")]:
    plt.plot(Bf, brillouin(3.5, 2 * 3.5 * MU_B * Bf / (K_B * Tk)), color=c, label=f"T = {Tk} K")
plt.xlabel("B (T)"); plt.ylabel("M / M_sat"); plt.title("paramagnet with J = 7/2 (Gd³⁺), g = 2")
plt.legend(); plt.show()

# %% [markdown]
# At room temperature a paramagnet is far from saturation even at 10 T. A ferromagnet is saturated at zero field –
# its moments are aligned by a much stronger interaction, the **exchange** (section 6). The ordered states:

# %%
rng = np.random.default_rng(0)
kinds = [("diamagnet\nno moments", None), ("paramagnet\nrandom", "rand"), ("ferromagnet\nparallel", "fm"),
         ("antiferromagnet\nantiparallel, compensated", "afm"), ("ferrimagnet\nantiparallel, not compensated", "fim")]
fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
for a, (title, kind) in zip(ax, kinds):
    xx, yy = np.meshgrid(np.arange(5), np.arange(4))
    if kind is None:
        a.plot(xx, yy, "o", color=GREY, ms=9)
    else:
        if kind == "rand":
            ang = rng.uniform(0, 2 * np.pi, xx.shape); L = np.ones(xx.shape)
        else:
            ang = np.full(xx.shape, np.pi / 2)
            if kind in ("afm", "fim"):
                ang[:, 1::2] = -np.pi / 2
            L = np.ones(xx.shape)
            if kind == "fim":
                L[:, 1::2] = 0.6
        a.quiver(xx, yy, L * np.cos(ang), L * np.sin(ang), np.where(np.sin(ang) >= 0, 0.0, 1.0), pivot="mid",
                 scale=6, width=0.02, cmap=plt.matplotlib.colors.ListedColormap([BLUE, ORANGE]), clim=(0, 1))
    a.set_title(title, fontsize=9); a.set_xlim(-0.7, 4.7); a.set_ylim(-0.7, 3.7); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
plt.tight_layout(); plt.show()

# %% [markdown]
# | Type | $\chi$ | Examples |
# |---|---|---|
# | diamagnetism | small, negative ($\sim -10^{-5}$) | Cu, Au, Bi, water, N₂, superconductors ($\chi = -1$) |
# | paramagnetism | small, positive ($10^{-6}$–$10^{-3}$) | Al, Pt, O₂, rare-earth salts |
# | ferromagnetism | large, positive, field-dependent; paramagnetic above $T_C$ | Fe, Co, Ni, Gd, Nd₂Fe₁₄B |
# | antiferromagnetism | small, positive; paramagnetic above $T_N$ | Cr, MnO, NiO |
# | ferrimagnetism | large, positive, field-dependent; paramagnetic above $T_C$ | magnetite Fe₃O₄, ferrites |

# %% [markdown]
# **Seeing it:** liquid nitrogen and liquid oxygen poured between the poles of a strong magnet. N₂ has all electrons
# paired (diamagnetic) and runs through; O₂ has two unpaired electrons in its antibonding π* orbitals
# (paramagnetic) and is held between the poles. (Video by Jefferson Lab on YouTube; it plays here only with an
# internet connection. Other demonstrations: Royal Institution, *Tales from the Prep Room – Paramagnetism of Liquid
# Oxygen*, `rz57PJToGEs`.)

# %%
from IPython.display import YouTubeVideo
YouTubeVideo("KcGEev8qulA", width=640, height=360)

# %% [markdown]
# **Estimate:** the susceptibility of liquid O₂ is about $\chi \approx 3.5\times10^{-3}$, that of liquid N₂ about
# $-4\times10^{-6}$. A magnet pulls on a liquid with the force density $f = \tfrac{\chi}{2\mu_0}\,\nabla (B^2)$.

# %%
B_gap, gap = 0.5, 0.01                                  # field near the pole gap (T) and length over which it falls (m)
rho_O2, rho_N2 = 1141, 807                              # densities of the liquids (kg/m^3)
for name, chi, rho_l in [("liquid O2", 3.5e-3, rho_O2), ("liquid N2", -4e-6, rho_N2)]:
    f_mag = chi / (2 * MU0) * B_gap**2 / gap             # N/m^3
    print(f"{name}: magnetic force density {f_mag:9.0f} N/m^3,  gravity {rho_l * 9.81:6.0f} N/m^3,"
          f"  ratio {f_mag / (rho_l * 9.81):+.3f}")

# %% [markdown]
# ## 6 Exchange and the Curie temperature
#
# The exchange interaction is a quantum effect (Pauli principle + Coulomb repulsion). In the Heisenberg model the
# energy of spins $\mathbf S_i$ on a lattice is
# $$\mathcal H = -2\sum_{i>j}\mathcal J_{ij}\,\mathbf S_i\cdot\mathbf S_j ;$$
# $\mathcal J > 0$ favours parallel spins (ferromagnet), $\mathcal J < 0$ antiparallel (antiferromagnet). The
# interaction decays quickly with distance, usually only the $z$ nearest neighbours matter. **Mean-field theory**
# replaces the neighbours by their average: every spin feels the field of all the others, which gives the
# self-consistent equation $m = B_S\!\left(\frac{3S}{S+1}\frac{m}{T/T_C}\right)$ for the reduced magnetization
# $m = M_s(T)/M_s(0)$, and
# $$T_C = \frac{2z\mathcal J S(S+1)}{3k_B}.$$
# Stronger exchange ⇒ higher Curie temperature. In a continuum, exchange becomes the energy density
# $A|\nabla\mathbf m|^2$ of micromagnetics (section 11); for iron $A \approx 2\times10^{-11}$ J/m.

# %%
def mean_field_m(t, S=0.5):
    if t >= 1:
        return 0.0
    f = lambda m: m - brillouin(S, 3 * S / (S + 1) * m / t)
    return brentq(f, 1e-6, 1.0) if f(1e-6) < 0 else 0.0


tt = np.linspace(0.01, 1.3, 300)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for S, c in [(0.5, BLUE), (3.5, ORANGE)]:
    ax[0].plot(tt, [mean_field_m(t, S) for t in tt], color=c, label=f"S = {S}")
ax[0].set_xlabel("T / T_C"); ax[0].set_ylabel("M_s(T) / M_s(0)"); ax[0].legend(); ax[0].set_title("mean-field theory")
TC = {"Ni": 627, "Fe": 1043, "Co": 1388, "Gd": 293, "Nd₂Fe₁₄B": 585, "Fe₃O₄": 858, "SmCo₅": 1020, "Sr ferrite": 733}
names = list(TC)
ax[1].barh(names, [TC[n] for n in names], color=BLUE)
ax[1].axvline(293, color=ORANGE, ls=":"); ax[1].text(300, 0, "room temperature", color=ORANGE, fontsize=8)
ax[1].set_xlabel("Curie temperature (K)")
plt.tight_layout(); plt.show()

# %% [markdown]
# Above $T_C$ the order is lost and the material becomes paramagnetic. Nd₂Fe₁₄B has a low $T_C$ (585 K) – one
# reason why NdFeB magnets in motors lose strength when hot and need dysprosium or terbium additions.

# %% [markdown]
# # Part II – From the energy budget to devices and data (lecture 2)
#
# The magnetization of a ferromagnet is the result of a competition of **four energy terms**. We switch them
# on one at a time – from a single uniformly magnetized particle (Stoner–Wohlfarth) to micromagnetic
# simulations of domains and walls – and then follow the material up to the devices: cores and their losses,
# transformers and generators, permanent magnets, and finally magnetic recording and memories.

# %% [markdown]
# ## 7 Energy terms and length scales
#
# **Predict:** an iron prism of 5 × 5 × 20 mm is to be simulated with cells of 1 nm. How many cells is that?
#
# A ferromagnet has four energy terms (per unit volume, $\mathbf m = \mathbf M/M_s$ is the unit magnetization):
#
# | Term | Energy density | Prefers |
# |---|---|---|
# | exchange | $A\,|\nabla \mathbf m|^2$ | parallel neighbouring moments |
# | anisotropy (uniaxial) | $-K\,(\mathbf m\cdot\mathbf u)^2$ | $\mathbf m$ along the easy axis $\mathbf u$ |
# | Zeeman | $-\mu_0 M_s\,\mathbf m\cdot\mathbf H$ | $\mathbf m$ along the applied field |
# | magnetostatic (stray field) | $-\tfrac12\mu_0 M_s\,\mathbf m\cdot\mathbf H_d$ | no magnetic charges, i.e. no stray field |
#
# Comparing them gives the characteristic lengths of a magnet:
# * **exchange length** $l_{ex}=\sqrt{2A/(\mu_0M_s^2)}$ – exchange versus stray field: below it the magnetization
#   cannot vary much;
# * **wall parameter** $\Delta=\sqrt{A/K}$ – exchange versus anisotropy: the domain wall is about $\pi\Delta$
#   wide.
#
# Micromagnetic simulations must resolve $l_{ex}$ (cells of a few nm), which limits them to (sub)micrometre
# samples.

# %%
l_ex = np.sqrt(2 * A_ex / (MU0 * Ms**2))
Delta = np.sqrt(A_ex / K1)
print(f"exchange length l_ex = {l_ex * 1e9:.2f} nm")
print(f"wall parameter Delta = sqrt(A/K) = {Delta * 1e9:.1f} nm, wall width pi*Delta = {np.pi * Delta * 1e9:.0f} nm")
n_cells = (5e-3 * 5e-3 * 20e-3) / (1e-9)**3
print(f"cells of 1 nm in a 5 x 5 x 20 mm prism: {n_cells:.1e}  (a large simulation today has ~1e8)")

# %% [markdown]
# So a millimetre-sized magnet can **not** be simulated cell by cell. Fortunately, the first two energy terms we
# study – the stray field and the Zeeman energy of a *uniformly* magnetized body – need no cells at all. The
# stray field of the iron prism and its demagnetizing factors were computed in section 3; now we add the applied
# field.

# %% [markdown]
# ## 8 Shape anisotropy and the Stoner–Wohlfarth model
#
# **Predict:** the prism is saturated along $+z$ and a field is applied along $-z$. At what field does the
# magnetization reverse? And what happens if the field is applied along $x$?
#
# If the magnetization stays **uniform** (one big “macrospin”), its energy density is
# $$e(\mathbf m) = -\mu_0M_s\,\mathbf H\cdot\mathbf m + \tfrac12\mu_0M_s^2\,(N_xm_x^2+N_ym_y^2+N_zm_z^2)
#   - K_u(\mathbf m\cdot\mathbf u)^2 .$$
# This is the **Stoner–Wohlfarth model** (1948). We apply the field in the $x$–$z$ plane at the angle $\psi$ from
# the long axis $z$; the magnetization then stays in this plane at an angle $\theta$, and with $N_x = N_y$ the
# energy reduces to one angle:
# $$e(\theta) = -\mu_0M_sH\cos(\theta-\psi) + K_{eff}\sin^2\theta,\qquad
#   K_{eff} = K_{sh} - K_u = \tfrac12\mu_0M_s^2(N_x-N_z) - K_u .$$
# For every field we let $\theta$ slide downhill into the **nearest** valley of $e(\theta)$ – not into the deepest
# one. This memory of the previous state is the origin of hysteresis.

# %%
theta = np.linspace(-np.pi, np.pi, 7201)[:-1]          # grid of magnetization angles (from the z axis)


def energy(H, psi, K_eff):
    """Energy density e(theta) (J/m^3) of the macrospin for the field H (A/m) at angle psi (rad) from z."""
    return -MU0 * Ms * H * np.cos(theta - psi) + K_eff * np.sin(theta)**2


def sw_loop(psi_deg, Hmax, K_eff, npts=801):
    """Stoner-Wohlfarth loop: field from +Hmax to -Hmax and back along the direction psi.
    Returns the fields and the magnetization angle; m along the field = cos(theta - psi)."""
    psi = np.deg2rad(psi_deg)
    Hs = np.concatenate([np.linspace(Hmax, -Hmax, npts), np.linspace(-Hmax, Hmax, npts)])
    i = np.argmin(np.abs(theta - psi))                   # start saturated along the field
    out = []
    for H in Hs:
        e = energy(H, psi, K_eff)
        while True:                                      # slide downhill to the nearest local minimum
            l, r = (i - 1) % len(theta), (i + 1) % len(theta)
            if e[l] < e[i]: i = l
            elif e[r] < e[i]: i = r
            else: break
        out.append(theta[i])
    return Hs, np.array(out)


H_K = (N[0] - N[2]) * Ms
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for H, c in zip([0, -0.5, -0.9, -1.0], ["k", BLUE, ORANGE, "r"]):
    e = energy(H * H_K, np.deg2rad(0.5), K_sh)
    ax[0].plot(np.rad2deg(theta), e / 1e3, color=c, label=f"H = {H} H_K")
ax[0].plot(0, energy(-0.9 * H_K, np.deg2rad(0.5), K_sh)[3600] / 1e3, "o", color=ORANGE)
ax[0].set_xlabel("θ (°)"); ax[0].set_ylabel("e (kJ/m³)"); ax[0].legend(fontsize=7)
ax[0].set_title("energy landscape, field along -z")
for a, ang in zip(ax[1:], [0.5, 45, 89.5]):
    Hs, th = sw_loop(ang, 2.0 * H_K, K_sh)
    a.plot(Hs / 1e3, np.cos(th - np.deg2rad(ang)), color=BLUE)
    a.set_title(f"field at {ang}° from the long axis"); a.set_xlabel("H (kA/m)"); a.set_ylabel("m along H")
    for s in (1, -1):
        a.axvline(s * H_K / 1e3, color=ORANGE, ls=":")
plt.tight_layout(); plt.show()
print(f"H_K = (Nx - Nz) Ms = 2 K_sh / (µ0 Ms) = {H_K / 1e3:.0f} kA/m  (orange dotted lines)")

# %% [markdown]
# * **Energy landscape:** at zero field there are two equal valleys, $\theta = 0$ and $180°$, separated by a barrier
#   $K_{sh}V$. A reverse field tilts the landscape; the magnetization stays in the shallower valley (orange dot)
#   until the valley disappears at $H = H_K$.
# * **Easy axis (field along $z$):** a square loop, the jump at $H_K = 2K_{eff}/(\mu_0M_s)$ (at slightly less if the
#   field is tilted by 0.5°, as here). Coercive field $H_c \approx H_K$.
# * **Hard axis (field along $x$):** no hysteresis; $\mathbf m$ rotates reversibly and saturates at $H_K$.
# * **In between:** a jump at a lower field than $H_K$.
#
# The switching field as a function of the field angle $\psi$ is the famous **Stoner–Wohlfarth astroid**,
# $h_{sw}(\psi) = H_{sw}/H_K = (\cos^{2/3}\psi + \sin^{2/3}\psi)^{-3/2}$. Let us check our model against it.

# %%
psi = np.array([1, 10, 20, 30, 45, 60, 70, 80, 89])
h_num = []
for a in psi:
    Hs, th = sw_loop(a, 1.2 * H_K, K_sh, npts=1201)
    mz = np.cos(th[:1201])                               # descending branch
    jump = np.argmax(np.abs(np.diff(mz)))
    h_num.append(-Hs[jump] / H_K)
p = np.linspace(0.5, 89.5, 200); t = np.deg2rad(p)
plt.figure(figsize=(5, 4))
plt.plot(p, (np.cos(t)**(2/3) + np.sin(t)**(2/3))**-1.5, color=ORANGE, label="astroid formula")
plt.plot(psi, h_num, "o", color=BLUE, label="numerical model")
plt.xlabel("field angle ψ from the easy axis (°)"); plt.ylabel("switching field / H_K"); plt.legend(); plt.show()

# %% [markdown]
# Above 45° the magnetization still jumps irreversibly, but the field where the loop crosses $m = 0$ (the coercive
# field) and the switching field are no longer the same – look at the 45° loop above.

# %% [markdown]
# ## 9 Magnetocrystalline anisotropy and magnetostriction
#
# Through the spin–orbit coupling the energy depends on the direction of $\mathbf M$ relative to the crystal axes.
# With the direction cosines $\alpha_i$ of $\mathbf M$:
# $$\text{uniaxial (hexagonal Co):}\quad e_a = K_1\sin^2\theta + K_2\sin^4\theta ,$$
# $$\text{cubic (Fe, Ni):}\quad e_a = K_1(\alpha_1^2\alpha_2^2 + \alpha_2^2\alpha_3^2 + \alpha_3^2\alpha_1^2)
#   + K_2\,\alpha_1^2\alpha_2^2\alpha_3^2 .$$
# The plots show $e_a$ as a surface: the distance from the centre grows with the energy, the **dimples are the
# easy axes**.

# %%
th, ph = np.meshgrid(np.linspace(0, np.pi, 91), np.linspace(0, 2 * np.pi, 181))
a1, a2, a3 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
cases = [("Co (hcp), uniaxial\nK1 = 4.1e5, K2 = 1.5e5 J/m³", 4.1e5 * np.sin(th)**2 + 1.5e5 * np.sin(th)**4),
         ("Fe (bcc), cubic, K1 = 4.8e4 J/m³\neasy ⟨100⟩", 4.8e4 * (a1**2 * a2**2 + a2**2 * a3**2 + a3**2 * a1**2)),
         ("Ni (fcc), cubic, K1 = −5.7e3 J/m³\neasy ⟨111⟩", -5.7e3 * (a1**2 * a2**2 + a2**2 * a3**2 + a3**2 * a1**2))]
fig = plt.figure(figsize=(14, 4.4))
for k, (title, e) in enumerate(cases):
    en = (e - e.min()) / (e.max() - e.min())
    R = 0.6 + en
    a = fig.add_subplot(1, 3, k + 1, projection="3d")
    a.plot_surface(R * a1, R * a2, R * a3, facecolors=plt.cm.viridis(en), rstride=2, cstride=2, linewidth=0)
    a.set_title(title, fontsize=9); a.set_box_aspect((1, 1, 1)); a.set_axis_off()
plt.tight_layout(); plt.show()

# %% [markdown]
# | Material | anisotropy | $K_1$ (J/m³) | anisotropy field $2K_1/(\mu_0M_s)$ |
# |---|---|---|---|
# | Fe | cubic | $4.8\times10^4$ | 56 kA/m |
# | Ni | cubic | $-5.7\times10^3$ | 19 kA/m |
# | Co | uniaxial | $4.1\times10^5$ | 0.6 MA/m |
# | Nd₂Fe₁₄B | uniaxial | $4.9\times10^6$ | 6 MA/m |
# | SmCo₅ | uniaxial | $1.7\times10^7$ | 32 MA/m |
#
# Soft materials need **small** anisotropy (Fe-Si, Ni-Fe near 80 % Ni, amorphous alloys), permanent magnets
# **large** uniaxial anisotropy (NdFeB, SmCo).
#
# **Magnetostriction.** The magnetization also deforms the lattice: a sample saturated along a direction changes
# its length by $\lambda = \Delta l/l$ of a few ppm (Fe −7, Ni −34, Co −62, Terfenol-D +2000; polycrystal values ×10⁻⁶).
# In AC devices this causes the 100 Hz **hum of transformers** and additional loss; the 6.5 % Si steel and Ni-Fe
# with ~80 % Ni have $\lambda \approx 0$. The volume effect (forced magnetostriction) compensates the thermal
# expansion of **Invar** (Fe-36 % Ni).

# %% [markdown]
# ### Try it: anisotropy, field and the hysteresis loop
#
# The two pictures of this and the previous section combined: a single-domain particle with a **uniaxial**
# anisotropy (easy axis $z$), an added **cubic** anisotropy $K_{cub}$ (cube axes along $x, y, z$) and a field $H$ at
# the angle $\psi$ from $z$ towards $x$. In units of $K_u$ the energy density is
# $$\frac{e(\mathbf m)}{K_u} = \sin^2\theta + \frac{K_{cub}}{K_u}\,(m_x^2m_y^2 + m_y^2m_z^2 + m_z^2m_x^2)
#   - 2\,\frac{H}{H_K}\,\mathbf m\cdot\hat{\mathbf h},\qquad H_K = \frac{2K_u}{\mu_0M_s}.$$
# The left panel shows this energy surface including the Zeeman term (the closest point to the centre is the
# deepest minimum), the right panel the loop. During the sweep $+H \to -H \to +H$ the magnetization (red arrow)
# stays in its **local** minimum until that minimum disappears – then it jumps: that is the hysteresis.
#
# Set the sliders and press **run field sweep** (rendering takes ~10 s, then the animation plays; the controls
# below it replay or step it). Try:
# * $K_{cub} = 0$, $\psi = 0°$: square loop, switching at exactly $H_K$; $\psi = 45°$: at $0.5\,H_K$; $\psi = 90°$: no
#   hysteresis at all (section 8);
# * $K_{cub}/K_u = +1$: the cubic term stiffens the easy axis, the switching field grows to $(1 + K_{cub}/K_u)H_K$;
# * $K_{cub}/K_u = -0.5$: the cubic term prefers the ⟨111⟩ diagonals – watch the magnetization leave the $x$–$z$
#   plane while it reverses.
#
# (If the widgets do not show, call `display(HTML(sweep_animation(1.0, 30)))` directly.)

# %%
import re, time
import ipywidgets as widgets
from IPython.display import display, HTML
from matplotlib import animation

H_MAX = 3.5          # sweep amplitude in units of H_K = 2K_u/(mu0 Ms); switches every case on the sliders


def energy(m, k, h):
    """Energy density / K_u of the unit vector(s) m: uniaxial (easy z) + k * cubic + Zeeman, h = H/H_K (vector)."""
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    return (1 - mz**2) + k * (mx**2 * my**2 + my**2 * mz**2 + mz**2 * mx**2) - 2 * (m @ h)


def relax(m, k, h, tol=1e-12, itmax=20000):
    """Follow the energy downhill on the unit sphere from m to the nearest local minimum (projected gradient)."""
    eta = 0.1 / (1 + abs(k) + np.linalg.norm(h))
    for _ in range(itmax):
        mx, my, mz = m
        g = np.array([2 * k * mx * (my**2 + mz**2), 2 * k * my * (mx**2 + mz**2),
                      -2 * mz + 2 * k * mz * (mx**2 + my**2)]) - 2 * h
        gt = g - (g @ m) * m                                 # tangential part of the gradient
        if gt @ gt < tol:
            break
        m = m - eta * gt
        m = m / np.linalg.norm(m)
    return m


def field_sweep(k, psi_deg, n=100, seed=1):
    """Quasi-static sweep +H_MAX -> -H_MAX -> +H_MAX along the direction psi (from z towards x)."""
    rng = np.random.default_rng(seed)
    u = np.array([np.sin(np.radians(psi_deg)), 0.0, np.cos(np.radians(psi_deg))])
    hs = np.r_[np.linspace(H_MAX, -H_MAX, n + 1), np.linspace(-H_MAX, H_MAX, n + 1)[1:]]
    m, ms = u.copy(), []
    for h in hs:
        m = relax(m + 1e-4 * rng.normal(size=3), k, h * u)  # tiny noise: no balancing on an unstable maximum
        ms.append(m)
    return hs, np.array(ms), u


TH, PH = np.meshgrid(np.linspace(0, np.pi, 31), np.linspace(0, 2 * np.pi, 61), indexing="ij")
DIRS = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)


def sweep_animation(k, psi_deg, every=3, interval=80, status=None):
    """Matplotlib animation: energy surface (distance from the centre = 0.6 + normalized energy, as above) with
    the magnetization (red) and the field (orange), and the hysteresis loop drawn point by point."""
    hs, ms, u = field_sweep(k, psi_deg)
    mpar = ms @ u                                           # magnetization component along the field
    idx = list(range(0, len(hs), every))
    fig = plt.figure(figsize=(11, 4.8), dpi=80)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], left=0.0, right=0.97, top=0.88, bottom=0.12, wspace=0.1)
    a3 = fig.add_subplot(gs[0], projection="3d")
    a2 = fig.add_subplot(gs[1])
    a2.set_xlim(-H_MAX * 1.05, H_MAX * 1.05); a2.set_ylim(-1.1, 1.1)
    a2.set_xlabel("H / H_K"); a2.set_ylabel("m along H"); a2.axhline(0, color="k", lw=0.5); a2.axvline(0, color="k", lw=0.5)
    loop, = a2.plot([], [], color=BLUE, lw=2)
    dot, = a2.plot([], [], "o", color="#d62728", ms=8)
    fig.suptitle(f"K_cub/K_u = {k:+.1f},  field at {psi_deg}° from the easy axis z (towards x)")

    def draw(j):
        i = idx[j]
        h = hs[i] * u
        e = energy(DIRS, k, h)
        en = (e - e.min()) / (e.max() - e.min() + 1e-12)
        R = 0.35 + en                                       # deep dimples: the minimum is the closest point
        a3.cla(); a3.set_axis_off(); a3.set_box_aspect((1, 1, 1), zoom=1.3)
        a3.set_xlim(-1.2, 1.2); a3.set_ylim(-1.2, 1.2); a3.set_zlim(-1.2, 1.2)
        a3.view_init(elev=12, azim=-75)
        a3.plot([0, 0], [0, 0], [-1.45, 1.45], color=GREY, lw=0.8, ls="--")
        a3.plot([-1.45, 1.45], [0, 0], [0, 0], color=GREY, lw=0.8, ls="--")
        a3.text(0, 0, 1.55, "z (easy)", color=GREY, fontsize=9, ha="center")
        a3.text(1.6, 0, 0, "x", color=GREY, fontsize=9)
        a3.plot_surface(R * DIRS[..., 0], R * DIRS[..., 1], R * DIRS[..., 2], facecolors=plt.cm.viridis(en),
                        rstride=1, cstride=1, linewidth=0, alpha=0.75, shade=True)
        m = ms[i]
        a3.quiver(0, 0, 0, *(1.5 * m), color="#d62728", lw=3, arrow_length_ratio=0.18)
        if abs(hs[i]) > 1e-9:
            a3.quiver(0, 0, 0, *(1.3 * h / H_MAX), color=ORANGE, lw=2.5, arrow_length_ratio=0.2)
        a3.text2D(0.03, 0.02, f"H = {hs[i]:+.2f} H_K\nred: m (energy minimum), orange: H", transform=a3.transAxes,
                  fontsize=10)
        loop.set_data(hs[:i + 1], mpar[:i + 1]); dot.set_data([hs[i]], [mpar[i]])
        if status is not None:
            status.value = f"rendering frame {j + 1} / {len(idx)} …"
        return loop, dot

    anim = animation.FuncAnimation(fig, draw, frames=len(idx), interval=interval, blit=False)
    with plt.rc_context({"animation.frame_format": "jpeg", "savefig.dpi": 80}):   # JPEG frames: ~5x smaller
        html = anim.to_jshtml(default_mode="once")
    plt.close(fig)
    name = re.search(r'onclick="(anim\w+)\.', html)
    if name:                                                # start playing as soon as it is shown
        html += (f"<script>setTimeout(function(){{ if (typeof {name.group(1)} !== 'undefined') "
                 f"{{ {name.group(1)}.play_animation(); }} }}, 500);</script>")
    return html


k_slider = widgets.FloatSlider(value=0.0, min=-0.8, max=2.0, step=0.1, description="K_cub / K_u",
                               continuous_update=False, style={"description_width": "110px"},
                               layout=widgets.Layout(width="420px"))
psi_slider = widgets.IntSlider(value=30, min=0, max=90, step=5, description="field angle (°)",
                               continuous_update=False, style={"description_width": "110px"},
                               layout=widgets.Layout(width="420px"))
run_button = widgets.Button(description="run field sweep", button_style="primary", icon="play")
status = widgets.Label("")
out = widgets.Output()


def run_sweep(_=None):
    run_button.disabled = True
    t0 = time.time()
    html = sweep_animation(k_slider.value, psi_slider.value, status=status)
    with out:
        out.clear_output(wait=True)
        display(HTML(html))
    status.value = f"done in {time.time() - t0:.0f} s – use the controls below the animation to replay or step"
    run_button.disabled = False


run_button.on_click(run_sweep)
display(widgets.VBox([out, widgets.HBox([widgets.VBox([k_slider, psi_slider]), widgets.VBox([run_button, status])])]))
run_sweep()                                                 # one sweep with the default settings

# %% [markdown]
# ## 10 Shape anisotropy versus magnetocrystalline anisotropy
#
# **Predict:** we add a uniaxial crystal anisotropy $K_u$ with its easy axis along $x$ – the *short* edge of the
# prism – so that it competes with the shape. What happens when $K_u = K_{sh}$?
#
# The two terms combine into $K_{eff} = K_{sh} - K_u$: positive means easy axis $z$, negative easy axis $x$ (the
# energy $K_{eff}\sin^2\theta = -K_{eff}\cos^2\theta + $ const). Iron's own anisotropy
# ($K_1 \approx 4.7\times10^4$ J/m³, and cubic, not uniaxial) is far too weak to compete with the shape of this
# prism ($K_{sh}\approx 6\times10^5$ J/m³), so we take a **model material** with $K_u = 0.5$, $1$ and $2\,K_{sh}$
# (such values occur e.g. in Co-based alloys). Along $y$ the crystal anisotropy does not help, so $\mathbf m$ stays
# in the $x$–$z$ plane and the one-angle model still holds.

# %%
fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for col, f in enumerate([0.5, 1.0, 2.0]):
    K_eff = K_sh - f * K_sh
    for row, (ang, lab) in enumerate([(0.5, "field along z (long axis)"), (89.5, "field along x (short axis)")]):
        Hs, th = sw_loop(ang, 1.5 * H_K, K_eff)
        ax[row, col].plot(Hs / 1e3, np.cos(th - np.deg2rad(ang)), color=BLUE if row == 0 else ORANGE)
        ax[row, col].set_title(f"Ku = {f} K_sh, {lab}", fontsize=10); ax[row, col].set_ylabel("m along H")
    easy = "z" if K_eff > 1 else ("none (easy plane x-z)" if abs(K_eff) <= 1 else "x")
    print(f"Ku = {f} K_sh:  K_eff = {K_eff:+.3g} J/m^3,  easy axis {easy},"
          f"  H_K = 2|K_eff|/(mu0 Ms) = {2 * abs(K_eff) / (MU0 * Ms) / 1e3:.0f} kA/m")
for a in ax[1]:
    a.set_xlabel("H (kA/m)")
plt.tight_layout(); plt.show()

# %% [markdown]
# * $K_u = 0.5\,K_{sh}$: the shape still wins, $z$ remains the easy axis, but the loop is half as wide.
# * $K_u = K_{sh}$: the two anisotropies cancel in the $x$–$z$ plane; the magnetization simply follows the field –
#   no hysteresis at all.
# * $K_u = 2\,K_{sh}$: the crystal anisotropy wins and the **short** axis becomes the easy axis.
#
# Along the easy axis the coercive field is always $H_c = 2|K_{eff}|/(\mu_0M_s)$.
#
# ### A problem
# For our iron prism (no crystal anisotropy) the Stoner–Wohlfarth model predicts $H_c \approx 580$ kA/m. A real
# bar of soft iron has $H_c \approx 0.1$ kA/m, and electrical steel about 0.05 kA/m – **ten thousand times less**.
# Something essential is missing: the magnetization of a real magnet does **not** stay uniform. To describe a
# non-uniform magnetization we need the last energy term – exchange – and with it, micromagnetics.

# %% [markdown]
# ## 11 Exchange: micromagnetics
#
# **Predict:** we simulate the same 1 : 1 : 4 prism, but only 5 × 5 × 20 nm in size, now with every
# nanometre-sized cell free to point in its own direction. Will it switch like the Stoner–Wohlfarth macrospin?
# And a prism 16 times larger?
#
# Micromagnetics keeps all four energy terms and lets $\mathbf m(\mathbf r)$ vary in space. The **exchange**
# energy $A|\nabla\mathbf m|^2$ penalizes non-uniform states; it is what keeps a small particle uniform – and what
# makes a domain wall cost energy. We use [magnum.np](https://gitlab.com/magnum.np/magnum.np), a finite-difference
# code written in PyTorch. In every field step the energy is minimized from the previous state (the same “nearest
# valley” rule as in section 8), now for thousands of coupled moments.

# %%
import importlib.util, os, subprocess, sys, time, csv, urllib.request
if importlib.util.find_spec("magnumnp") is None:                       # on Colab: install magnum.np (~1 min)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "magnumnp"], check=True)
import torch
import magnumnp
from magnumnp import (Mesh, State, DemagField, ExchangeField, UniaxialAnisotropyField, ExternalField,
                      MinimizerBB, LLGSolver, set_log_level)
set_log_level(40)

REPO = "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/comsol_datasets/data/magnetism"


def fetch(name):
    """Path of a stored result: from a local copy of the repository, or downloaded from GitHub."""
    for d in ("../comsol_datasets/data/magnetism", "comsol_datasets/data/magnetism"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    if not os.path.exists(name):
        urllib.request.urlretrieve(f"{REPO}/{name}", name)
    return name


def mm_prism_loop(w, cell, Ku=0.0, axis=(1, 0, 0), field_dir_deg=0.5, Hmax=700e3, dH=20e3):
    """Descending branch of the loop of a w x w x 4w prism (long axis z) computed with magnum.np.
    The field lies in the x-z plane at field_dir_deg from z. Returns H (A/m) and <m> along the field."""
    n = int(round(w / cell))
    state = State(Mesh((n, n, 4 * n), (w / n, w / n, w / n)))
    state.material = {"Ms": Ms, "A": A_ex, "Ku": Ku, "Ku_axis": list(axis)}
    psi = np.deg2rad(field_dir_deg)
    d = np.array([np.sin(psi), 0, np.cos(psi)])
    state.m = state.Constant(list(d))                                 # saturated along the field
    ext = ExternalField(state.Constant([0, 0, 0]))
    terms = [DemagField(), ExchangeField(), ext] + ([UniaxialAnisotropyField()] if Ku else [])
    minimizer = MinimizerBB(terms)
    Hs, mpar = np.arange(Hmax, -Hmax - 1, -dH), []
    for H in Hs:
        ext.h = state.Constant(list(H * d))
        minimizer.minimize(state, maxiter=50000, dm_tol=1.0)
        mpar.append(float((state.m.mean(dim=(0, 1, 2)).cpu().numpy() * d).sum()))
    return Hs, np.array(mpar)


t0 = time.time()
# (title, Ku, field angle from z, field angle from the easy axis for SW, K_eff)
cases = [("no crystal anisotropy, field along z", 0.0, 0.5, 0.5, K_sh),
         ("Ku = 2 K_sh along x, field along x", 2 * K_sh, 89.5, 0.5, K_sh)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (lab, Ku, ang, ang_sw, K_eff) in zip(ax, cases):
    Hs, mpar = mm_prism_loop(5e-9, 1e-9, Ku=Ku, field_dir_deg=ang)
    a.plot(Hs / 1e3, mpar, "o", ms=4, color=BLUE, label="magnum.np, 5 × 5 × 20 nm")
    a.plot(-Hs / 1e3, -mpar, "o", ms=4, color=BLUE, mfc="none")            # ascending branch by symmetry
    Hsw, th = sw_loop(ang_sw, 700e3, K_eff)
    a.plot(Hsw / 1e3, np.cos(th - np.deg2rad(ang_sw)), color=ORANGE, label="Stoner–Wohlfarth")
    a.set_title(lab, fontsize=10); a.set_xlabel("H (kA/m)"); a.set_ylabel("m along H"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"computed in {time.time() - t0:.0f} s")

# %% [markdown]
# A prism of 5 × 5 × 20 nm switches **as the Stoner–Wohlfarth macrospin**: it is only a few exchange lengths
# wide, so exchange keeps it uniform, and the simple model of sections 8–10 is confirmed by a completely different
# method. (Open symbols: the ascending branch, obtained by symmetry. With $K_u = 2K_{sh}$ along $x$ the effective
# anisotropy along $x$ equals $K_{sh}$, so the loop looks the same as without crystal anisotropy along $z$.)
#
# Now the size. The next results were computed in advance (larger prisms take minutes to hours): loops of 1 : 1 : 4
# prisms from 5 to 80 nm, the field 0.5° from the long axis, no crystal anisotropy.

# %%
sizes = [5, 10, 20, 40, 80]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
Hc = []
for w, c in zip(sizes, plt.cm.Blues(np.linspace(0.4, 1, len(sizes)))):
    L = np.loadtxt(fetch(f"prism_loop_w{w}nm.csv"), delimiter=",", skiprows=1)
    ax[0].plot(L[:, 0] / 1e3, L[:, 3], "-", color=c, label=f"{w} × {w} × {4 * w} nm")
    Hc.append(-L[np.argmax(L[:, 3] < 0), 0])
H_sw0 = 0.94 * H_K                                     # SW switching field for a field tilted by 0.5°
ax[0].axvline(-H_sw0 / 1e3, color=ORANGE, ls=":", label="Stoner–Wohlfarth")
ax[0].set_xlabel("H (kA/m)"); ax[0].set_ylabel("<m_z>"); ax[0].legend(fontsize=8); ax[0].set_title("descending branch")
ax[1].semilogx(sizes, np.array(Hc) / 1e3, "o-", color=BLUE, label="magnum.np")
ax[1].axhline(H_sw0 / 1e3, color=ORANGE, ls=":", label="Stoner–Wohlfarth")
ax[1].set_xlabel("width w of the prism (nm)"); ax[1].set_ylabel("switching field (kA/m)"); ax[1].legend()
plt.tight_layout(); plt.show()
for w, h in zip(sizes, Hc):
    print(f"w = {w:3d} nm   H_sw = {h / 1e3:5.0f} kA/m   ({h / H_sw0:.2f} of Stoner–Wohlfarth)")

# %% [markdown]
# Above a width of a few exchange lengths the switching field drops: the magnetization no longer rotates as a whole.
# The picture shows the magnetization of the 40 nm prism in the last field step before the switch: the core still
# points along $+z$, but near the surface and the end faces – where the stray field is strongest – $\mathbf m$ has
# already turned far away from the axis. From there the reversal spreads through the prism. The 80 nm prism no
# longer switches in one jump at all: it reverses gradually through a vortex state, at about 20 kA/m.

# %%
snap = np.load(fetch("prism_snap_w40nm.npz"))["m"]              # (nx, ny, nz, 3)
nx, ny, nz, _ = snap.shape
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
sl = snap[:, ny // 2]                                              # plane y = 0: (x, z)
im = ax[0].imshow(sl[:, :, 2], origin="lower", cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax[0].quiver(np.arange(nz)[::3], np.arange(nx)[::2], sl[::2, ::3, 2], sl[::2, ::3, 0], scale=25)
ax[0].set_xlabel("z (cells)"); ax[0].set_ylabel("x (cells)"); ax[0].set_title("plane y = 0, colour = m_z")
ax[1].plot(np.arange(nz), sl[nx // 2, :, 2], color=BLUE, label="m_z on the axis")
ax[1].plot(np.arange(nz), sl[0, :, 2], color=ORANGE, label="m_z at the surface")
ax[1].set_xlabel("z (cells)"); ax[1].legend()
plt.colorbar(im, ax=ax[0]); plt.tight_layout(); plt.show()

# %% [markdown]
# ### Why domains form
#
# **Predict:** without any applied field, which state has the lowest energy – the uniform one, two antiparallel
# domains separated by a wall, or a vortex in which $\mathbf m$ circulates around the long axis?
#
# A uniform magnet pays the stray-field energy $\tfrac12\mu_0M_s^2N_z$ per unit volume – independent of its size. A
# non-uniform state reduces the stray field but pays exchange (and, with anisotropy, wall) energy, which per unit
# volume *decreases* with size (a wall is an area, the stray field fills a volume). Below, the relaxation was started
# from the three states for prisms of growing size (stored results); we compare the energy densities after the
# energy minimization.

# %%
rows = list(csv.reader(open(fetch("prism_states.csv"))))
head, rows = rows[0], rows[1:]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, c in zip(["uniform", "two_domain", "vortex"], [BLUE, ORANGE, "gray"]):
    r = [x for x in rows if x[1] == k]
    w = np.array([float(x[0]) for x in r]); e = np.array([float(x[4]) for x in r]); mz = np.array([float(x[7]) for x in r])
    ax[0].plot(w, e / 1e3, "o-", color=c, label=f"start: {k}")
    ax[1].plot(w, np.abs(mz), "o-", color=c, label=f"start: {k}")
ax[0].axhline(0.5 * MU0 * Ms**2 * N[2] / 1e3, color="k", ls=":", label="uniform, ½µ0Ms²Nz")
ax[0].set_xlabel("width w of the 1:1:4 prism (nm)"); ax[0].set_ylabel("energy density after relaxation (kJ/m³)")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("width w (nm)"); ax[1].set_ylabel("|<m_z>| after relaxation"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# * Small prisms: every starting state relaxes back to the uniform one – the prism is **single-domain**; a wall
#   would cost more exchange energy than it saves in stray-field energy.
# * Large prisms: the non-uniform states survive and have a **lower** energy than the uniform state; the
#   stray-field energy is reduced at the price of exchange. In the largest prism even the uniform start does not
#   survive – it breaks up by itself (right panel). This is why every large ferromagnet breaks up into
#   domains – and why it can be magnetized and demagnetized by small fields: moving walls is much easier than
#   rotating the whole magnetization against the shape anisotropy.
#
# The crossing is the **single-domain size** – about 30 nm for this iron prism (the vortex is the first state to beat
# the uniform one; the two-domain state needs more room for its wall and wins only above about 50 nm). For a hard
# magnet such as Nd-Fe-B, with a large $K$, the single-domain size is a few hundred nanometres.

# %% [markdown]
# ## 12 Domains
#
# A uniformly magnetized body has a large stray-field energy. Splitting it into domains with opposite
# magnetization reduces the stray field, at the cost of the **domain-wall energy** $\sigma_w = 4\sqrt{AK}$ per area;
# closure domains remove the stray field completely.

# %%
def block(a, x0, w, h, arrows, ec=BLUE):
    a.add_patch(Rectangle((x0, 0), w, h, fill=False, ec=ec, lw=1.5))
    n = len(arrows)
    for k, s in enumerate(arrows):
        xc = x0 + (k + 0.5) * w / n
        if k:
            a.plot([x0 + k * w / n] * 2, [0, h], color=GREY, lw=0.8, ls="--")
        a.add_patch(FancyArrow(xc, h / 2 - s * 0.3 * h, 0, s * 0.6 * h, width=0.04, head_width=0.16,
                               head_length=0.12, length_includes_head=True, color=BLUE))


def charges(a, x0, w, h, arrows):
    n = len(arrows)
    for k, s in enumerate(arrows):
        xs = np.linspace(x0 + k * w / n + 0.05, x0 + (k + 1) * w / n - 0.05, 3)
        a.plot(xs, [h + 0.06] * 3, "+" if s > 0 else "_", color="r", ms=6)
        a.plot(xs, [-0.06] * 3, "_" if s > 0 else "+", color="r", ms=6)


fig, a = plt.subplots(figsize=(14, 3.2))
x = 0
for arrows, lab in [([1], "single domain:\nlarge stray field"), ([1, -1], "two domains"),
                    ([1, -1, 1, -1], "four domains:\nstray field ~1/4")]:
    block(a, x, 1.2, 2, arrows); charges(a, x, 1.2, 2, arrows); a.text(x, -0.55, lab, fontsize=9); x += 2
# closure domains: two main domains + triangular caps
x0, w, h = x, 1.6, 2
a.add_patch(Rectangle((x0, 0), w, h, fill=False, ec=BLUE, lw=1.5))
for p in ([(x0, 0), (x0 + w / 2, w / 4), (x0 + w, 0)], [(x0, h), (x0 + w / 2, h - w / 4), (x0 + w, h)]):
    a.plot(*zip(*p), color=GREY, lw=0.8, ls="--")
a.plot([x0 + w / 2] * 2, [w / 4, h - w / 4], color=GREY, lw=0.8, ls="--")
for xc, s in [(x0 + w / 4, 1), (x0 + 3 * w / 4, -1)]:
    a.add_patch(FancyArrow(xc, h / 2 - s * 0.4, 0, s * 0.8, width=0.04, head_width=0.16, head_length=0.12,
                           length_includes_head=True, color=BLUE))
for yc, s in [(0.12, 1), (h - 0.12, -1)]:
    a.add_patch(FancyArrow(x0 + w / 2 - s * 0.3, yc, s * 0.6, 0, width=0.03, head_width=0.12, head_length=0.1,
                           length_includes_head=True, color=ORANGE))
a.text(x0, -0.55, "closure domains:\nno stray field", fontsize=9)
a.set_xlim(-0.3, x0 + w + 0.3); a.set_ylim(-0.8, 2.3); a.set_aspect("equal"); a.axis("off")
plt.show()

# %% [markdown]
# **How wide are the domains?** For a plate of thickness $L$ with stripe domains of width $D$ (uniaxial material,
# magnetization perpendicular to the surface; Kittel's model) the energy per surface area is
# $$\frac{E}{\text{area}} = \underbrace{\sigma_w\frac{L}{D}}_{\text{walls}}
#   + \underbrace{0.136\,\mu_0M_s^2D}_{\text{stray field}}
#   \quad\Rightarrow\quad D_{opt} = \sqrt{\frac{\sigma_wL}{0.136\,\mu_0M_s^2}} \propto \sqrt L .$$

# %%
A_Fe, K_Fe = 2.1e-11, 4.8e4
sig_w = 4 * np.sqrt(A_Fe * K_Fe)
L = np.logspace(-8, -2, 200)
D = np.sqrt(sig_w * L / (0.136 * MU0 * Ms_Fe**2))
plt.figure(figsize=(6.5, 4))
plt.loglog(L * 1e6, D * 1e6, color=BLUE, label="optimal stripe width")
plt.loglog(L * 1e6, L * 1e6, ":", color=GREY, label="D = L")
plt.xlabel("plate thickness L (µm)"); plt.ylabel("domain width D (µm)"); plt.legend()
plt.title(f"Fe-like parameters, σ_w = 4√(AK) = {sig_w * 1e3:.1f} mJ/m²"); plt.show()
print(f"wall energy {sig_w * 1e3:.1f} mJ/m^2; domain width in a 0.35 mm sheet: {np.interp(0.35e-3, L, D) * 1e6:.0f} µm")

# %% [markdown]
# Below some tens of nanometres $D$ would exceed $L$: small particles stay **single-domain** (section
# 11). The model applies to a magnetization **perpendicular** to the plate (e.g. a uniaxial magnet or a
# recording film). In electrical steel the easy axes lie in the plane of the sheet, the surface carries almost no
# magnetic charge, and the domains are much wider – 0.1–1 mm, set by the grains and the closure domains at their
# boundaries; the walls between them are only ~0.1 µm thick (section 13).

# %% [markdown]
# ## 13 The domain wall
#
# **Predict:** the wall width is set by exchange (wants a slow rotation) and anisotropy (wants $\mathbf m$ along the
# easy axis). If $A$ is doubled, does the wall get wider or narrower, and by how much? And if $K$ is doubled?
#
# From now on we study a single wall in an **iron ribbon** of 300 × 5 × 1 nm with the easy axis along its length
# $x$. The wall is a rotation of $\mathbf m$ from $+x$ to $-x$; along the ribbon it is effectively one-dimensional.
# Minimizing $\int [A(\partial_x\theta)^2 + K\sin^2\theta]\,dx$ gives the classical profile
# $$m_x = -\tanh\frac{x-x_0}{\Delta},\qquad m_y = \frac{1}{\cosh((x-x_0)/\Delta)},\qquad \Delta = \sqrt{A/K},$$
# and a wall energy $4\sqrt{AK}$ per unit area.
#
# **Note:** in this ribbon the stray field is switched off. The reference model (COMSOL Multiphysics with the
# micromagnetics plugin of Fudan University) has no stray field either; the ribbon is then a clean 1D model of a
# wall *inside* a large crystal, where the stray field is already minimized by the domain structure. The rotation in
# the $x$–$y$ plane is just the starting choice; with the stray field included, it would decide the wall type
# (Bloch or Néel).

# %%
def ribbon_state(K, A, L=300e-9, cell=2e-9, x0=150e-9, d0=30e-9, alpha=1.0, m0=None):
    """1D ribbon (no stray field), cells of 2 nm, with a wall at x0 (or the start m0, shape (n, 3));
    K (J/m^3) and A (J/m) may be arrays along x."""
    n = int(round(L / cell))
    x = (np.arange(n) + 0.5) * cell
    state = State(Mesh((n, 1, 1), (cell, 5e-9, 1e-9)))
    Kt = state.Constant(0.0); Kt[:, 0, 0, 0] = torch.tensor(np.broadcast_to(K, n).copy(), dtype=Kt.dtype)
    At = state.Constant(0.0); At[:, 0, 0, 0] = torch.tensor(np.broadcast_to(A, n).copy(), dtype=At.dtype)
    state.material = {"Ms": Ms, "A": At, "Ku": Kt, "Ku_axis": [1, 0, 0], "alpha": alpha}
    if m0 is None:
        m0 = np.stack([-np.tanh((x - x0) / d0), 1 / np.cosh((x - x0) / d0), 0 * x], -1)
    state.m = state.Constant([1, 0, 0]); state.m[:, 0, 0, :] = torch.tensor(m0, dtype=state.m.dtype)
    return state, x


from scipy.optimize import curve_fit
tanh_wall = lambda x, x0, d: -np.tanh((x - x0) / d)
comsol5 = np.load(fetch("ribbon_wall_width_comsol.npz"))
fK, fA = [0.5, 1, 2], [0.5, 1, 2]
res = np.zeros((3, 3, 3))                      # (K, A) -> theory, magnum.np, COMSOL
t0 = time.time()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for i, a in enumerate(fK):
    for j, b in enumerate(fA):
        K, A = a * K1, b * A_ex
        state, x = ribbon_state(K, A)
        MinimizerBB([ExchangeField(), UniaxialAnisotropyField()]).minimize(state, maxiter=50000, dm_tol=0.1)
        m = state.m[:, 0, 0, :].cpu().numpy()
        (x0, d), _ = curve_fit(tanh_wall, x * 1e9, m[:, 0], p0=[150, 20])
        pc = comsol5[f"w{i}{j}"][-1]                 # COMSOL, last time step: (x, component)
        (x0c, dc), _ = curve_fit(tanh_wall, comsol5["x_nm"], pc[:, 0], p0=[150, 20])
        res[i, j] = np.sqrt(A / K) * 1e9, d, dc
        if j == 1:
            c = plt.cm.Blues(0.4 + 0.3 * i)
            ax[0].plot(x * 1e9, m[:, 0], color=c, label=f"m_x, K = {a} K1 (magnum.np)")
            ax[0].plot(x * 1e9, m[:, 1], "--", color=c)
            ax[0].plot(comsol5["x_nm"][::3], pc[::3, 0], "o", ms=3, color=ORANGE)
ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("m"); ax[0].set_title("walls for A = A_Fe (dashed: m_y; dots: COMSOL)")
ax[0].legend(fontsize=7)
ax[1].plot(res[..., 0].ravel(), res[..., 1].ravel(), "o", color=BLUE, label="magnum.np (fit)")
ax[1].plot(res[..., 0].ravel(), res[..., 2].ravel(), "x", ms=9, color=ORANGE, label="COMSOL (fit)")
ax[1].plot([8, 45], [8, 45], "k:", label="Δ = sqrt(A/K)")
ax[1].set_xlabel("sqrt(A/K) (nm)"); ax[1].set_ylabel("fitted Δ (nm)"); ax[1].legend()
plt.tight_layout(); plt.show()
print(f"computed in {time.time() - t0:.0f} s")
print("Delta (nm)                K = 0.5 K1    K1     2 K1")
for j, b in enumerate(fA):
    print(f"A = {b:3} A_Fe  theory     " + "  ".join(f"{res[i, j, 0]:6.2f}" for i in range(3)))
    print(f"              magnum.np  " + "  ".join(f"{res[i, j, 1]:6.2f}" for i in range(3)))
    print(f"              COMSOL     " + "  ".join(f"{res[i, j, 2]:6.2f}" for i in range(3)))

# %% [markdown]
# Both codes give $\Delta = \sqrt{A/K}$: doubling $A$ widens the wall by $\sqrt2$, doubling $K$ narrows it by
# $\sqrt2$. Only the widest wall feels the ends of the ribbon. Below, the rotation of $\mathbf m$ through the wall,
# drawn as arrows along the ribbon (reference iron: $\Delta = 21$ nm, so the wall is about $\pi\Delta \approx 66$ nm
# wide).

# %%
state, x = ribbon_state(K1, A_ex)
MinimizerBB([ExchangeField(), UniaxialAnisotropyField()]).minimize(state, maxiter=50000, dm_tol=0.1)
m = state.m[:, 0, 0, :].cpu().numpy()
fig, ax = plt.subplots(figsize=(13, 2.2))
k = slice(30, 120, 3)
ax.quiver(x[k] * 1e9, 0 * x[k], m[k, 0], m[k, 1], m[k, 0], cmap="coolwarm", scale=30, width=0.004, pivot="mid")
ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xlabel("x (nm)"); ax.set_title("m in the plane of the ribbon (colour: m_x)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 14 Wall motion
#
# **Predict:** the domain on the left points along $+x$, the one on the right along $-x$. A field along $+x$ is
# applied. Which way does the wall move, and does it move faster when the damping is larger or smaller?
#
# The magnetization obeys the **Landau–Lifshitz–Gilbert (LLG) equation**,
# $$\frac{\partial\mathbf m}{\partial t} = -\gamma\mu_0\,\mathbf m\times\mathbf H_{eff}
#   + \alpha\,\mathbf m\times\frac{\partial\mathbf m}{\partial t},$$
# a precession around the effective field plus damping (Gilbert constant $\alpha$). In our 1D ribbon (no stray field,
# so no hard axis) a field $H$ along the easy axis moves the wall with a constant velocity
# $$v = \frac{\alpha}{1+\alpha^2}\,\gamma\mu_0 H\,\Delta ,$$
# fastest for $\alpha = 1$. (With a hard axis, e.g. from the stray field of a thin film, the velocity is
# $\gamma\mu_0H\Delta/\alpha$ below the *Walker field* and drops above it – a topic for a more advanced course.)
#
# **The ends of the ribbon.** The ends are free surfaces ($\partial\mathbf m/\partial x = 0$). Without a stray field
# a wall *gains* energy by leaving the ribbon through a free end – a wall started next to an end, or pushed against
# one, simply disappears. We therefore make both ends (30 nm each) of **harder iron** ($K = 20\,K_0$), standing for
# the harder material around a soft grain. As in a real sample, where reversal starts at an edge, the wall is
# **nucleated at an end**: the left end is magnetized along $+x$, the iron along $-x$, and a 10 nm seed between them
# along $+y$. We first let this start **relax at zero field** (the wall forms and settles next to the hard end), then
# switch on the field.

# %%
magnumnp.common.constants.gamma = 2.21e5                  # gyromagnetic ratio gamma*mu0 (m/(A s)), as in COMSOL


def ribbon_K(K0=K1, eps=0.0, lam=64e-9, kend=20, cell=3e-9, L=300e-9):
    """Anisotropy along the ribbon: pinning modulation eps (section 15) and harder ends (30 nm, kend*K0)."""
    x = (np.arange(int(round(L / cell))) + 0.5) * cell
    K = K0 * (1 - eps * np.cos(2 * np.pi * (x - 150e-9) / lam))
    ends = (x < 30e-9) | (x > 270e-9)
    K[ends] = kend * K0
    return x, K, ends


def end_seed(x):
    """Start: left end along +x, seed along +y on 30-40 nm, the rest along -x."""
    m0 = np.zeros((len(x), 3)); m0[:, 0] = -1
    m0[x < 30e-9] = (1, 0, 0)
    m0[(x >= 30e-9) & (x < 40e-9)] = (0, 1, 0)
    return m0


def run_llg(state, H, t_run, dt=0.5e-9):
    """Integrate the LLG equation for t_run with a constant field H (A/m) along x; returns times, m(t, x, 3)."""
    terms = [ExchangeField(), UniaxialAnisotropyField()] + ([ExternalField(state.Constant([H, 0, 0]))] if H else [])
    llg = LLGSolver(terms)
    t_start = float(state.t)
    times, profiles = [], []
    for t in t_start + np.arange(dt, t_run + dt / 2, dt):
        llg.step(state, t - float(state.t))
        times.append(t); profiles.append(state.m[:, 0, 0, :].cpu().numpy().copy())
    return np.array(times), np.array(profiles)


def wall_position(x, mx):
    """Position of the wall: where m_x crosses zero (linear interpolation)."""
    cross = np.where(np.sign(mx[:-1]) != np.sign(mx[1:]))[0]
    if len(cross) == 0:
        return np.nan                                     # no wall in the ribbon
    i = cross[0]
    return x[i] - mx[i] * (x[i + 1] - x[i]) / (mx[i + 1] - mx[i])


# Cells of 3 nm: coarser cells allow longer time steps, the wall (about 67 nm) still spans 20 cells.
t0 = time.time()
x, K6, ends6 = ribbon_K()
state, x = ribbon_state(K6, A_ex, cell=3e-9, m0=end_seed(x))
t_relax, H6 = 4e-9, 8e3
t_r, p_r = run_llg(state, 0.0, t_relax)                   # 4 ns at zero field: the wall forms and relaxes
t_d, p_d = run_llg(state, H6, 12e-9)                      # then 12 ns at 8 kA/m along +x
times = np.concatenate([[0.0], t_r, t_d])
prof = np.concatenate([end_seed(x)[None], p_r, p_d])
pos = np.array([wall_position(x[~ends6], p[~ends6, 0]) for p in prof])
print(f"computed in {time.time() - t0:.0f} s")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for k in range(0, len(times), 4):
    ax[0].plot(x * 1e9, prof[k, :, 0], color=plt.cm.Blues(0.3 + 0.7 * k / len(times)), label=f"t = {times[k] * 1e9:.0f} ns")
for a in (0, 270):
    ax[0].axvspan(a, a + 30, color="gray", alpha=0.3)
ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("m_x"); ax[0].set_title("grey: harder iron"); ax[0].legend(fontsize=7)
ax[1].plot(times * 1e9, pos * 1e9, "o-", color=BLUE)
ax[1].axvspan(0, t_relax * 1e9, color="gray", alpha=0.2); ax[1].text(0.3, 250, "H = 0", fontsize=8)
ax[1].text(t_relax * 1e9 + 0.3, 250, f"H = {H6 / 1e3:.0f} kA/m", fontsize=8)
ax[1].axhline(270, color="gray", ls=":")
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("wall position (nm)")
plt.tight_layout(); plt.show()

# %% [markdown]
# At zero field the seed turns into a wall that moves a few tens of nanometres away from the hard end and slows
# down: the hard end repels it (a wall in harder iron would cost more energy), but in uniform iron nothing else holds
# it. The field then pushes it through the iron at a constant velocity until the hard end on the right stops it –
# the ribbon is reversed except for the right end. The same as an animation (in Colab the player appears below the
# cell):

# %%
from matplotlib import animation
fig, ax = plt.subplots(figsize=(10, 2.6))
for a in (0, 270):
    ax.axvspan(a, a + 30, color="gray", alpha=0.3)
line_x, = ax.plot(x * 1e9, prof[0, :, 0], color=BLUE, label="m_x")
line_y, = ax.plot(x * 1e9, prof[0, :, 1], color=ORANGE, label="m_y")
txt = ax.set_title("")
ax.set_ylim(-1.1, 1.1); ax.set_xlabel("x (nm)"); ax.legend(loc="lower left")


def frame(k):
    line_x.set_ydata(prof[k, :, 0]); line_y.set_ydata(prof[k, :, 1])
    txt.set_text(f"t = {times[k] * 1e9:.1f} ns, H = {0 if times[k] <= t_relax + 1e-12 else H6 / 1e3:.0f} kA/m")
    return line_x, line_y, txt


anim = animation.FuncAnimation(fig, frame, frames=len(times), interval=150)
try:
    from IPython.display import HTML, display
    display(HTML(anim.to_jshtml()))
except Exception:
    pass
plt.close(fig)

# %% [markdown]
# **Velocity versus field and damping.** We measure the velocity from the wall position and compare it with the
# formula. To measure the motion in uniform iron only, the (relaxed tanh) wall now starts at 100 nm, far enough from
# the hard end that the end no longer pushes it.

# %%
t0 = time.time()
Hlist = np.array([2e3, 4e3, 8e3])
plt.figure(figsize=(6, 4))
for alpha, c in [(1.0, BLUE), (0.2, ORANGE)]:
    v = []
    for H in Hlist:
        state, x = ribbon_state(K6, A_ex, cell=3e-9, x0=100e-9, d0=np.sqrt(A_ex / K1), alpha=alpha)
        tv, pv = run_llg(state, H, 3e-9)
        posv = np.array([wall_position(x, p[:, 0]) for p in pv])
        v.append(np.polyfit(tv[1:], posv[1:], 1)[0])
    Hf = np.linspace(0, 8.5e3, 50)
    plt.plot(Hf / 1e3, alpha / (1 + alpha**2) * 2.21e5 * Hf * np.sqrt(A_ex / K1), "-", color=c)
    plt.plot(Hlist / 1e3, v, "o", color=c, label=f"α = {alpha}")
plt.xlabel("H (kA/m)"); plt.ylabel("wall velocity (m/s)"); plt.legend(title="dots: magnum.np, lines: formula")
plt.show()
print(f"computed in {time.time() - t0:.0f} s")

# %% [markdown]
# The simulated velocities follow the formula; the small deviation at low fields comes from the short runs (the wall
# needs about a nanosecond to reach its steady velocity) and the 3 nm cells. Note the time scale: a wall crosses a
# 100 nm grain in about 10 ns, while a 50 Hz field changes over 20 ms. For the wall, the field of a transformer is
# **static** – the hysteresis loop at 50 Hz is a quasi-static loop.

# %% [markdown]
# ## 15 Rotation versus wall propagation: two coercive fields
#
# **Predict:** the same iron ribbon is reversed once by a uniform rotation (no wall present) and once by pushing an
# existing wall through it. Which needs the larger field, and by roughly how much?
#
# We compare two ways to reverse the ribbon, each by a field held constant for 20 ns (much longer than the
# nanosecond time scale of the wall, so the result is practically static):
# * **Rotation:** a uniform ribbon, magnetized along $-x$ (tilted by 1° so that the field exerts a torque), no wall.
#   The Stoner–Wohlfarth model predicts switching at $H_K = 2K/(\mu_0M_s)$.
# * **Wall propagation:** a ribbon with a wall that has to travel through a landscape of **pinning sites** – a
#   periodic modulation of the anisotropy, $K(x) = K_0[1-\varepsilon\cos(2\pi(x-150\,\mathrm{nm})/64\,\mathrm{nm})]$,
#   standing for the defects, grain boundaries and stress of a real material. The wall is nucleated at the left
#   end exactly as in section 14 (harder-iron ends, seed along $+y$) and relaxes for 5 ns at zero field; it settles in
#   the first pinning site at 86 nm. Then the field is switched on and held for 20 ns.
#
# The ribbon is switched when the average $m_x$ of the iron between the ends changes sign. COMSOL computed 75 such
# runs (3 anisotropies for the rotation, 4 combinations of anisotropy and pinning strength for the wall, each at
# 6–12 fields; damping $\alpha = 1$). In the wall runs the time axis includes the 5 ns relaxation (grey).

# %%
sw = list(csv.DictReader(open(fetch("ribbon_switching_comsol.csv"))))
dat7 = np.load(fetch("ribbon_switching_comsol.npz"))


def coercive_field(rows):
    """Midpoint between the largest field that did not switch and the smallest field that did."""
    H = np.array([float(r["H_A_per_m"]) for r in rows]); s = np.array([int(r["switched"]) for r in rows])
    lo, hi = H[s == 0].max() if (s == 0).any() else 0.0, H[s == 1].min() if (s == 1).any() else np.nan
    return 0.5 * (lo + hi), lo, hi


groups = {}
for r in sw:
    if r["mechanism"]:
        key = (r["mechanism"], float(r["HK_A_per_m"]), float(r["pinning_eps"]))
        groups.setdefault(key, []).append(r)
print("mechanism    H_K (kA/m)  eps   Hc (kA/m)  [largest not switched, smallest switched]")
Hc7 = {}
for key in sorted(groups):
    hc, lo, hi = coercive_field(groups[key])
    Hc7[key] = hc
    hc_txt = f"{hc / 1e3:6.1f}" if np.isfinite(hc) else f"> {lo / 1e3:.1f}"
    print(f"{key[0]:11s}  {key[1] / 1e3:6.1f}     {key[2]:.1f}   {hc_txt:>6s}     [{lo / 1e3:.1f}, {hi / 1e3:.1f}]")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for key, c in zip(sorted(groups), plt.cm.tab10(np.arange(len(groups)))):
    rows7 = sorted(groups[key], key=lambda r: float(r["H_A_per_m"]))
    H = np.array([float(r["H_A_per_m"]) for r in rows7]); m_end = np.array([float(r["mx_iron_end"]) for r in rows7])
    ax[0].semilogx(H / 1e3, m_end, "o-", color=c, label=f"{key[0]}, H_K = {key[1] / 1e3:.0f} kA/m, ε = {key[2]}")
ax[0].set_xlabel("field held for 20 ns (kA/m)"); ax[0].set_ylabel("<m_x> of the iron after 20 ns"); ax[0].legend(fontsize=7, loc="upper left")
ref = sorted(groups[("wall", 43700.0, 0.4)], key=lambda r: float(r["H_A_per_m"]))
for r, c in zip(ref, plt.cm.Blues(np.linspace(0.3, 1, len(ref)))):
    cid = r["case_id"]
    ax[1].plot(dat7[cid + "_t_ns"], dat7[cid + "_mx_iron"], color=c, label=f"{float(r['H_A_per_m']) / 1e3:.0f} kA/m")
ax[1].axvspan(0, 5, color="gray", alpha=0.2)
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("<m_x> of the iron"); ax[1].set_title("wall, H_K = 43.7 kA/m, ε = 0.4 (grey: H = 0)")
ax[1].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.show()

# %% [markdown]
# The wall does not glide smoothly: it **jumps from one pinning site to the next** and stops wherever the local
# restoring force of the pinning exceeds the push of the field. The space–time map of $m_x$ along the ribbon shows
# this stop-and-go motion (in a real material it is heard as *Barkhausen noise*).

# %%
r_dep = [r for r in ref if int(r["switched"])][0]                    # the smallest field that switches
r_pin = [r for r in ref if not int(r["switched"])][-1]               # the largest field that does not
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for a, r in zip(ax, [r_pin, r_dep]):
    cid = r["case_id"]; m7 = dat7[cid + "_m"]
    a.pcolormesh(dat7["x_nm"], dat7[cid + "_t_ns"], m7[:, :, 0], cmap="coolwarm", vmin=-1, vmax=1, shading="auto")
    for xp in 150 + 64 * np.arange(-2, 2):
        a.axvline(xp, color="k", ls=":", lw=0.8)                       # minima of K: the pinning sites
    a.axvspan(0, 30, color="gray", alpha=0.4); a.axvspan(270, 300, color="gray", alpha=0.4)
    a.axhline(5, color="k", lw=0.8)                                    # the field is switched on at 5 ns
    a.set_title(f"H = {float(r['H_A_per_m']) / 1e3:.0f} kA/m ({'switched' if int(r['switched']) else 'pinned'})")
    a.set_xlabel("x (nm)")
ax[0].set_ylabel("t (ns)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Your turn: reproduce a COMSOL point with magnum.np
#
# The function below builds the pinned ribbon, nucleates the wall at the left end and relaxes it for 5 ns at zero
# field (once), then applies a constant field for 20 ns. Run it for a field below and above the COMSOL coercive
# field of the reference case and compare $\langle m_x\rangle(t)$ (this takes a minute or two per run on the Colab
# CPU).

# %%
def relaxed_pinned_ribbon(K0=K1, eps=0.4, lam=64e-9, kend=20, t_relax=5e-9, cell=3e-9):
    """Pinned ribbon with the wall nucleated at the left end, relaxed at zero field; returns x, K, ends, m, <m_x>(t)."""
    x, K, ends = ribbon_K(K0, eps, lam, kend, cell)
    state, x = ribbon_state(K, A_ex, cell=cell, m0=end_seed(x))
    t, p = run_llg(state, 0.0, t_relax)
    hist = [[0.0, end_seed(x)[~ends, 0].mean()]] + [[ti, pi[~ends, 0].mean()] for ti, pi in zip(t, p)]
    return x, K, ends, p[-1], np.array(hist)


def pinned_ribbon_run(H, relaxed, t_end=20e-9, cell=3e-9):
    """Constant field H (A/m) for t_end, starting from the relaxed wall; returns (t, <m_x> of the iron)."""
    x, K, ends, m_rel, hist = relaxed
    state, x = ribbon_state(K, A_ex, cell=cell, m0=m_rel)
    t, p = run_llg(state, H, t_end)
    return np.concatenate([hist, np.stack([t + hist[-1, 0], p[:, ~ends, 0].mean(1)], 1)])


t0 = time.time()
relaxed = relaxed_pinned_ribbon()
print(f"wall after the relaxation at x = {wall_position(relaxed[0], relaxed[3][:, 0]) * 1e9:.0f} nm")
plt.figure(figsize=(7, 4))
for r, c in zip([r_pin, r_dep], [BLUE, ORANGE]):
    H = float(r["H_A_per_m"])
    res7 = pinned_ribbon_run(H, relaxed)
    plt.plot(res7[:, 0] * 1e9, res7[:, 1], color=c, label=f"magnum.np, {H / 1e3:.0f} kA/m")
    plt.plot(dat7[r["case_id"] + "_t_ns"], dat7[r["case_id"] + "_mx_iron"], "--", color=c, label=f"COMSOL, {H / 1e3:.0f} kA/m")
plt.axvspan(0, 5, color="gray", alpha=0.2)
plt.xlabel("t (ns)"); plt.ylabel("<m_x> of the iron"); plt.legend(fontsize=8); plt.show()
print(f"computed in {time.time() - t0:.0f} s")

# %% [markdown]
# ### Two hysteresis loops
#
# Collecting the coercive fields gives the quasi-static loops of the ribbon for the two mechanisms. (A field that
# changes within nanoseconds would widen the loop further, because the wall lags behind the field; at 50 Hz this
# dynamic widening is negligible.)

# %%
hc_rot = Hc7[("rotation", 43700.0, 0.0)]
hc_wall = Hc7[("wall", 43700.0, 0.4)]
fig, ax = plt.subplots(figsize=(7, 4.2))
for hc, c, lab in [(hc_rot, ORANGE, "rotation (no wall)"), (hc_wall, BLUE, "wall propagation (ε = 0.4)")]:
    ax.plot([-80e3, -hc, -hc, 80e3], [-1, -1, 1, 1], color=c, lw=2, label=f"{lab}: Hc = {hc / 1e3:.1f} kA/m")
    ax.plot([80e3, hc, hc, -80e3], [1, 1, -1, -1], color=c, lw=2)
ax.set_xticks(np.arange(-80e3, 81e3, 20e3)); ax.set_xticklabels([f"{v:.0f}" for v in np.arange(-80, 81, 20)])
ax.set_xlabel("H (kA/m)"); ax.set_ylabel("m_x"); ax.legend(fontsize=8); plt.show()
print(f"Hc(rotation) / Hc(wall) = {hc_rot / hc_wall:.1f}")

# %% [markdown]
# * **Rotation** switches the ribbon close to $H_K = 43.7$ kA/m, as Stoner–Wohlfarth predicts. (Just above $H_K$ the
#   rotation starts very slowly, so within 20 ns the switch is seen only a few per cent above $H_K$.)
# * **Wall propagation** needs only the field that tears the wall out of the pinning sites – the **depinning field**.
#   It depends on the pinning strength $\varepsilon$ and on $H_K$, but is many times smaller than $H_K$. Just above
#   the depinning field the wall moves slowly, in jumps (the 5 kA/m map above): it is still on its way after 20 ns.
# * **Pinning sites without a wall** (the *nucleation* runs: the uniform ribbon with the same anisotropy modulation)
#   do not help: the ribbon does not switch even at $H_K$. The weak spots ($K = 0.6\,K_0$) are only about 32 nm
#   long, too short for a reversed region to form against the exchange coupling to the stiffer neighbours. Weak
#   spots lower the coercive field only when they are larger than the wall width – which is why real magnets
#   reverse from edges, surface defects or already existing walls.
#
# Domain walls are the reason why soft magnetic materials are soft. Without walls, even iron would be a (weak)
# permanent magnet.

# %% [markdown]
# ## 16 The hysteresis loop
#
# Magnetizing a demagnetized sample first moves the domain walls (reversibly, then irreversibly in jumps – the
# Barkhausen effect), then rotates the magnetization of the remaining domains into the field. The loop is
# characterized by:
# * the **saturation** flux density $B_s$ (or polarization $J_s = \mu_0 M_s$);
# * the **remanence** $B_r$ at $H = 0$;
# * the **coercive field** $H_c$ where $B = 0$ ($H_{cJ}$ where $M = 0$);
# * the **initial permeability** $\mu_i = \frac1{\mu_0}\lim_{H\to0}\frac BH$ and the **maximum permeability**
#   $\mu_{max} = \max\frac{B}{\mu_0H}$ on the initial curve;
# * the **area** $\oint H\,\mathrm dB$ – the energy dissipated per cycle and volume (hysteresis loss).
#
# A simple model loop (the branches are shifted copies of one anhysteretic curve):

# %%
def model_loop(Ms, Hc, a, Hmax, n=801):
    """Branches M(H) = Ms*tanh((H +- Hc)/a) and an initial curve M = Ms*tanh(H/a)*(1 - exp(-H/Hc)) (schematic)."""
    H = np.linspace(-Hmax, Hmax, n)
    down, up = Ms * np.tanh((H + Hc) / a), Ms * np.tanh((H - Hc) / a)
    Hi = np.linspace(0, Hmax, n)
    init = Ms * np.tanh(Hi / a) * (1 - np.exp(-Hi / Hc))
    return H, down, up, Hi, init


Ms_s, Hc_s, a_s = 1.2e6, 60.0, 80.0          # a soft steel (A/m)
H, Md, Mu, Hi, Mi = model_loop(Ms_s, Hc_s, a_s, 600)
Bd, Bu, Bi = MU0 * (H + Md), MU0 * (H + Mu), MU0 * (Hi + Mi)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(H, Bd, color=BLUE); ax[0].plot(H, Bu, color=BLUE)
ax[0].plot(Hi, Bi, "--", color=GREY, label="initial curve")
Br = np.interp(0, H, Bd); Hc_B = -H[np.argmin(np.abs(Bd))]
for (xp, yp, lab) in [(0, Br, "B_r"), (-Hc_B, 0, "−H_c"), (H[-1], Bd[-1], "B_s")]:
    ax[0].plot(xp, yp, "o", color=ORANGE); ax[0].annotate(lab, (xp, yp), textcoords="offset points", xytext=(6, -12))
ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
ax[0].set_xlabel("H (A/m)"); ax[0].set_ylabel("B (T)"); ax[0].legend(); ax[0].set_title("soft steel (model)")
mur = Bi[1:] / (MU0 * Hi[1:])
ax[1].plot(Hi[1:], mur, color=BLUE)
k = np.argmax(mur)
ax[1].plot(Hi[k], mur[k], "o", color=ORANGE); ax[1].annotate("µ_max", (Hi[k], mur[k]), xytext=(8, 0),
                                                          textcoords="offset points")
ax[1].annotate("µ_i", (Hi[1], mur[0]), xytext=(8, 0), textcoords="offset points")
ax[1].set_xlabel("H (A/m)"); ax[1].set_ylabel("µ_r = B/(µ0 H) on the initial curve")
plt.tight_layout(); plt.show()
trapz = getattr(np, "trapezoid", None) or np.trapz
loss = trapz(Bd - Bu, H)                    # area between the descending and the ascending branch
print(f"Br = {Br:.2f} T, Hc = {Hc_B:.0f} A/m, mu_max = {mur[k]:.0f};  loop area = {loss:.0f} J/m^3 per cycle "
      f"-> at 50 Hz: {loss * 50 / 7650:.1f} W/kg")

# %% [markdown]
# **Soft and hard.** The same loop shape with a different coercive field spans the whole range of magnetic
# materials – from transformer cores to permanent magnets:

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, (name, Ms_, Hc_, aa, Hmax) in zip(ax, [("soft: Ni-Fe, H_c ≈ 1 A/m", 0.6e6, 1.0, 2.0, 15),
                                             ("electrical steel, H_c ≈ 50 A/m", 1.6e6, 50.0, 60.0, 600),
                                             ("hard: NdFeB, H_cJ ≈ 1 MA/m", 1.0e6, 1.0e6, 1.5e5, 2.5e6)]):
    H, Md, Mu, _, _ = model_loop(Ms_, Hc_, aa, Hmax)
    a.plot(H / 1e3 if Hmax > 1e4 else H, MU0 * (H + Md), color=BLUE)
    a.plot(H / 1e3 if Hmax > 1e4 else H, MU0 * (H + Mu), color=BLUE)
    a.set_title(name, fontsize=10); a.set_xlabel("H (kA/m)" if Hmax > 1e4 else "H (A/m)"); a.set_ylabel("B (T)")
    a.axhline(0, color="k", lw=0.6); a.axvline(0, color="k", lw=0.6)
plt.tight_layout(); plt.show()

# %% [markdown]
# Note the field axes: the coercive fields differ by **six orders of magnitude**. Why real materials have coercive
# fields so much smaller than their anisotropy field was the subject of sections 10–15: the domain walls
# move at fields far below the field needed to rotate the magnetization coherently.

# %% [markdown]
# ## 17 From the loop to the loss
#
# Let us put the coercive fields we met side by side, together with real materials.

# %%
ladder = [("Stoner–Wohlfarth, iron prism (shape)", 0.94 * H_K),
          ("rotation, iron ribbon (H_K)", hc_rot),
          ("wall depinning, ribbon, ε = 0.4", hc_wall),
          ("wall depinning, ribbon, ε = 0.2", Hc7.get(("wall", 43700.0, 0.2), np.nan)),
          ("soft iron (typical)", 80.0),
          ("non-oriented electrical steel (typical)", 50.0),
          ("grain-oriented steel / nanocrystalline (typical)", 5.0)]
plt.figure(figsize=(8, 3.6))
plt.barh([l for l, _ in ladder][::-1], [v for _, v in ladder][::-1],
         color=[ORANGE] * 3 + [BLUE] * 4)                          # reversed order: real materials first
plt.xscale("log"); plt.xlabel("coercive field (A/m)"); plt.title("blue: this notebook, orange: real materials")
plt.tight_layout(); plt.show()

# %% [markdown]
# Even the pinned wall needs a field about a hundred times larger than electrical steel. Real soft magnetic
# materials are better because:
# * their walls are pinned much more weakly (large, clean grains; the pinning in our ribbon – a 40 % change of the
#   anisotropy every 64 nm – is very strong);
# * thermal activation helps the wall over the pinning barriers (micromagnetics here is at 0 K);
# * materials are engineered for it: Si lowers the anisotropy and the magnetostriction of iron, nanocrystalline
#   alloys average the anisotropy out over many tiny grains.
#
# **From the loop to the loss.** The area of the hysteresis loop is the energy lost per cycle and unit volume,
# $W = \oint \mu_0 H\,dM$; for a square loop $W \approx 4\mu_0M_sH_c$. At the frequency $f$ the **hysteresis loss**
# is $P = fW$. Device models use the empirical Steinmetz/Bertotti term $P_h = k_h f B^{\beta}$ fitted to
# measurements – for example the generator model in this repository uses $k_h = 1000$ W/(m³·Hz·T$^\beta$) and
# $\beta = 1.5$.

# %%
f, B = 50.0, 1.5
W_ribbon = 4 * MU0 * Ms * hc_wall
W_steel = 4 * MU0 * Ms * 50.0
W_bertotti = 1000 * B**1.5
rho = 7650                                                 # density of electrical steel (kg/m^3)
for lab, W in [("pinned wall in our ribbon", W_ribbon), ("electrical steel, Hc = 50 A/m", W_steel),
               ("Bertotti term of the generator model, B = 1.5 T", W_bertotti)]:
    print(f"{lab:48s}  W = {W:9.0f} J/m^3 per cycle  ->  {f * W / rho:8.2f} W/kg at 50 Hz")
print(f"\nThe Bertotti term corresponds to a square loop with Hc = {W_bertotti / (4 * MU0 * Ms):.0f} A/m")

# %% [markdown]
# The hysteresis loss of a real core lies between the ideal electrical steel and our strongly pinned ribbon – and
# its coefficients are **measured**, not computed: micromagnetics explains *why* soft materials are soft and what
# controls their coercivity, while the device simulation uses a measured B(H) curve and fitted loss coefficients.
# Section 18 adds the losses caused by eddy currents, and section 19 takes these material models up to the device
# level: the transformer and the permanent-magnet generator.

# %% [markdown]
# ## 18 Core losses
#
# In an AC field the core loss per volume is separated into three parts (Bertotti):
# $$p = \underbrace{k_h f B^{\beta}}_{\text{hysteresis}}
#   + \underbrace{\frac{\pi^2\sigma d^2}{6} f^2B^2}_{\text{classical eddy current}}
#   + \underbrace{k_e (fB)^{1.5}}_{\text{excess}}\quad(\text{W/m}^3),$$
# $f$ the frequency, $B$ the peak flux density, $\sigma$ the conductivity and $d$ the lamination thickness.
# The **hysteresis** term is the loop area times $f$ (sections 16–17); the **classical** eddy
# current loss comes from Faraday's law in a sheet with a uniform field; the **excess** loss from the eddy currents
# around moving domain walls (section 14). The specific loss (W/kg) is $p$ divided by the density; data
# sheets give e.g. $p_{1.5/50}$, the loss at 1.5 T and 50 Hz.
#
# The parameters below are chosen to give the typical loss of a 0.35 mm non-oriented electrical steel
# (≈ 2.4 W/kg at 1.5 T, 50 Hz).

# %%
rho_m, sig = 7650.0, 2.0e6          # density (kg/m^3), conductivity of 3 % Si steel (S/m)
kh, beta, ke = 109.0, 2.0, 5.9       # hysteresis and excess coefficients (SI)


def core_loss(f, B, d, sigma=sig):
    """Hysteresis, classical and excess loss (W/kg)."""
    ph = kh * f * B**beta
    pc = np.pi**2 * sigma * d**2 / 6 * f**2 * B**2
    pe = ke * (f * B)**1.5
    return ph / rho_m, pc / rho_m, pe / rho_m


f = np.logspace(0, 3.5, 200)
ph, pc, pe = core_loss(f, 1.5, 0.35e-3)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].stackplot(f, ph / f, pc / f, pe / f, colors=[BLUE, ORANGE, GREY], alpha=0.8,
                labels=["hysteresis", "classical eddy", "excess"])
ax[0].set_xscale("log"); ax[0].set_xlabel("f (Hz)"); ax[0].set_ylabel("loss per cycle (J/kg)")
ax[0].legend(loc="upper left", fontsize=8); ax[0].set_title("loss separation, 1.5 T, d = 0.35 mm")
for dd, c in [(0.1e-3, BLUE), (0.35e-3, ORANGE), (0.65e-3, GREY)]:
    ax[1].loglog(f, sum(core_loss(f, 1.5, dd)), color=c, label=f"d = {dd * 1e3:.2f} mm")
ax[1].set_xlabel("f (Hz)"); ax[1].set_ylabel("total loss at 1.5 T (W/kg)"); ax[1].legend(fontsize=8)
ax[1].set_title("thinner sheets for higher frequency")
mu_r = 5000
delta = np.sqrt(2 / (2 * np.pi * f * MU0 * mu_r * sig))
ax[2].loglog(f, delta * 1e3, color=BLUE, label=f"skin depth, µr = {mu_r}, σ = 2 MS/m")
for dd in (0.1, 0.35, 0.65):
    ax[2].axhline(dd, color=GREY, ls=":"); ax[2].text(1.2, dd * 1.08, f"{dd} mm", fontsize=8, color=GREY)
ax[2].set_xlabel("f (Hz)"); ax[2].set_ylabel("δ (mm)"); ax[2].legend(fontsize=8)
ax[2].set_title("δ = √(2 / (ω µ σ))")
plt.tight_layout(); plt.show()
print("at 1.5 T, 50 Hz, d = 0.35 mm: hysteresis %.2f, classical %.2f, excess %.2f W/kg" % core_loss(50, 1.5, 0.35e-3))

# %% [markdown]
# * At 50 Hz the **hysteresis** loss dominates; the eddy-current terms grow faster with $f$ and take over in the
#   kHz range (motors of electric cars run at 400–1000 Hz, converters at tens of kHz).
# * The classical loss scales with $\sigma d^2$: hence **thin, insulated laminations** of **silicon steel**
#   (3 % Si raises the resistivity from 0.1 to 0.5 µΩm). The formula assumes a uniform field across the sheet,
#   i.e. $d$ smaller than the skin depth $\delta$.
# * At high frequency only materials with a very high resistivity remain useful: **ferrites** (insulators),
#   amorphous and nanocrystalline ribbons (≈ 20 µm thick), soft magnetic composites (insulated powder).

# %% [markdown]
# ## 19 Devices: the transformer and the generator
#
# **Predict:** the iron of a generator is replaced by one that is ten times *less* permeable. Does the induced
# voltage drop ten times, a little, or not at all? And what happens to the iron loss?
#
# We now leave the nanometre scale. A device simulation does not resolve domains; the material enters through two
# **measured** curves – the anhysteretic B(H) curve and a loss formula – and Maxwell's equations are solved for the
# whole machine. Their low-frequency (magneto-quasi-static) form is
# $$\nabla\times\mathbf H = \mathbf J,\qquad \nabla\cdot\mathbf B = 0,\qquad
#   \nabla\times\mathbf E = -\frac{\partial\mathbf B}{\partial t},$$
# with the constitutive laws $\mathbf B = \mu_0\mu_r(|\mathbf H|)\,\mathbf H$ in the iron (the nonlinear B(H)
# curve), $\mathbf B = \mu_0\mu_{rec}\mathbf H + \mathbf B_r$ in a permanent magnet and $\mathbf J = \sigma\mathbf E$
# in conductors. With the vector potential, $\mathbf B = \nabla\times\mathbf A$, they become one equation,
# $$\sigma\frac{\partial\mathbf A}{\partial t} + \nabla\times\bigl(\nu(B)\,\nabla\times\mathbf A\bigr)
#   = \mathbf J_{coil} + \nabla\times(\nu\,\mathbf B_r),\qquad \nu = 1/(\mu_0\mu_r),$$
# which COMSOL solves with finite elements. A winding of $N$ turns couples the field to the electric circuit by
# Faraday's law, $u = Ri + \mathrm d\psi/\mathrm dt$ with the flux linkage $\psi \approx N\Phi$.
#
# **Iron loss.** Laminated iron is treated as non-conducting in the field solution (the laminations block the eddy
# currents); the loss is computed afterwards from the local flux density, harmonic by harmonic, with the
# **Bertotti formula** (W/m³, amplitude $B$ at frequency $f$):
# $$p = \underbrace{k_h f B^{\beta}}_{\text{hysteresis (section 17)}}
#   + \underbrace{\frac{\pi^2\sigma d^2}{6} f^2 B^2}_{\text{classical eddy currents}}
#   + \underbrace{8\sqrt{\sigma G S V_0}\,(fB)^{3/2}}_{\text{excess (wall motion)}} .$$
# The first term is the loop area of section 17 times $f$; the third comes from the eddy currents around *moving domain
# walls* (sections 14–15) – the micromagnetic picture survives in the device model as three fitted constants.
#
# For each device we have a **COMSOL dataset** (computed in advance), we compare it with a **second, independent
# simulation method** that you run here, and then we train a **neural network (DNN) surrogate** on the COMSOL data.

# %%
import os, time, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU0 = 4e-7 * np.pi
BLUE, ORANGE = "#1f4e8c", "#e07b10"
DATA = "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/comsol_datasets/data"


def fetch_data(sub, name):
    """Path of a dataset file: from a local copy of the repository, or downloaded from GitHub."""
    for d in (f"../comsol_datasets/data/{sub}", f"comsol_datasets/data/{sub}"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    if not os.path.exists(name):
        urllib.request.urlretrieve(f"{DATA}/{sub}/{name}", name)
    return name


# %% [markdown]
# ### 19.1 The transformer
#
# The device is the *E-core transformer* of the COMSOL Application Library: two E-cores of soft iron (outline
# 80 × 70 mm, depth 20 mm, centre leg 20 mm, windows 20 × 50 mm), primary and secondary winding with 300 turns each on
# the centre leg, a sinusoidal source $v = V_{ac}\sin(2\pi ft)$ with a series resistance $R_p = 100\ \Omega$ on the
# primary side and a load resistance $R_s$ on the secondary side. COMSOL solves the **3D transient** field problem
# coupled to the circuit for 2.5 periods; the iron loss (Steinmetz term $k_h fB^{1.5}$, $k_h = 1000$) is the cycle
# average over the last period. We varied $V_{ac}$ (5–100 V), $f$ (25–200 Hz) and $R_s$ (10 Ω – 100 kΩ),
# sampling the three-dimensional box with a quasi-random (Sobol) sequence: every run takes 2–5 minutes, so the
# samples must be spread evenly.

# %%
tr = pd.read_csv(fetch_data("transformer_3d", "transformer_cases.csv"))
trw = np.load(fetch_data("transformer_3d", "transformer_waveforms.npz"))
print(f"{len(tr)} COMSOL runs, total solve time {tr.solve_time_s.sum() / 3600:.1f} h")
print(tr[["Vac_V", "f_Hz", "Rload_ohm", "V2_rms_V", "I1_rms_A", "P_core_W", "P_load_W", "B_max_T"]]
      .describe().loc[["min", "50%", "max"]].round(3).to_string())

# %% [markdown]
# **Second method: the magnetic circuit.** Instead of the 3D field we apply Ampère's law along the centre line of
# the core: the flux $\Phi$ of the centre leg splits into the two outer legs, every branch $k$ (length $l_k$, cross
# section $A_k$) needs the field $H(B_k)$ of the B(H) curve,
# $$N_p i_p + N_s i_s = \sum_k H\!\left(\frac{\Phi_k}{A_k}\right) l_k ,\qquad
#   v(t) = R_p i_p + N_p\frac{\mathrm d\Phi}{\mathrm dt},\qquad 0 = R_s i_s + N_s\frac{\mathrm d\Phi}{\mathrm dt}.$$
# Eliminating the currents gives one ordinary differential equation for $\Phi(t)$. The iron follows an analytic
# curve $H = B/(\mu_0\mu_i)\,[1 + (B/B_0)^p]$ ($\mu_i = 1200$, $B_0 = 1.25$ T, $p = 8$) that approximates the soft iron
# of the COMSOL model; leakage flux and the coil resistance are neglected.

# %%
class CircuitTransformer:
    """Nonlinear magnetic circuit of the E-core transformer coupled to the primary and secondary circuit."""

    def __init__(self, Vac=25.0, freq=50.0, Rs=1e4, Np=300, Ns=300, Rp=100.0, depth=0.02, leg_c=0.02, leg_o=0.01,
                 yoke=0.01, win_w=0.02, win_h=0.05, mu_i=1200.0, B0=1.25, p=8.0, kh=1000.0, beta=1.5):
        self.__dict__.update(locals()); del self.__dict__["self"]
        l_leg, l_yoke = win_h + yoke, 0.5 * leg_c + win_w + 0.5 * leg_o
        self.branches = [(l_leg, leg_c * depth, 1.0),          # centre leg: (length, area, share of the flux)
                         (2 * l_yoke, yoke * depth, 0.5),      # yoke halves, per side
                         (l_leg, leg_o * depth, 0.5)]          # outer leg, per side
        self.volume = depth * ((leg_c + 2 * win_w + 2 * leg_o) * (win_h + 2 * yoke) - 2 * win_w * win_h)
        path = sum(l * a * (1 if s == 1 else 2) for l, a, s in self.branches)
        self.vscale = self.volume / path                       # the centre-line path counts the corners twice

    def H_iron(self, B):
        return B / (MU0 * self.mu_i) * (1 + np.abs(B / self.B0) ** self.p)

    def mmf(self, phi):
        return sum(self.H_iron(phi * s / a) * l for l, a, s in self.branches)

    def run(self, periods=2.5, points=400):
        """From rest, source v = Vac sin(2 pi f t) as in COMSOL; returns the quantities of the last period."""
        w = 2 * np.pi * self.freq
        k = self.Np + self.Rp * self.Ns ** 2 / (self.Np * self.Rs)
        rhs = lambda t, phi: (self.Vac * np.sin(w * t) - self.Rp * self.mmf(phi) / self.Np) / k
        n = int(periods * points)
        t = np.linspace(0, periods / self.freq, n + 1)
        phi = solve_ivp(rhs, (0, t[-1]), [0.0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-12).y[0]
        dphi = rhs(t, phi)
        i_s = -self.Ns * dphi / self.Rs
        i_p = (self.mmf(phi) - self.Ns * i_s) / self.Np
        last = slice(n - points, n)                              # the last full period
        P_core = 0.0
        for l, a, s in self.branches:                           # Steinmetz loss, harmonic by harmonic
            B = phi[last] * s / a
            amp = 2 / points * np.abs(np.fft.rfft(B))[1:points // 2]
            fh = self.freq * np.arange(1, points // 2)
            P_core += l * a * (1 if s == 1 else 2) * self.vscale * (self.kh * fh * amp ** self.beta).sum()
        rms = lambda x: np.sqrt(np.mean(x[last] ** 2))
        return dict(t=(t[last] - t[n - points]) * self.freq, i_p=i_p[last], u_s=self.Ns * dphi[last],
                    B=phi[last] / (self.leg_c * self.depth), V2_rms_V=rms(self.Ns * dphi), I1_rms_A=rms(i_p),
                    P_core_W=P_core, P_load_W=self.Rs * np.mean(i_s[last] ** 2))


t0 = time.time()
circ = pd.DataFrame([CircuitTransformer(Vac=r.Vac_V, freq=r.f_Hz, Rs=r.Rload_ohm).run() for r in tr.itertuples()])
print(f"magnetic circuit for all {len(tr)} cases: {time.time() - t0:.1f} s")

# %% [markdown]
# Waveforms over one period for three runs – a lightly loaded one, a heavily loaded one and one with a saturated
# core (dashed: magnetic circuit):

# %%
picks = [tr.B_max_T.idxmin(), tr.Rload_ohm.idxmin(), tr.B_max_T.idxmax()]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for a, i in zip(ax, picks):
    r, cid = tr.loc[i], tr.case_id[i]
    run = CircuitTransformer(Vac=r.Vac_V, freq=r.f_Hz, Rs=r.Rload_ohm).run()
    a.plot(trw[cid + "_t"], trw[cid + "_V2"], color=BLUE, label="u_s (V), COMSOL")
    a.plot(run["t"], run["u_s"], "--", color=BLUE, label="u_s, circuit")
    a2 = a.twinx()
    a2.plot(trw[cid + "_t"], trw[cid + "_I1"] * 1e3, color=ORANGE, label="i_p (mA), COMSOL")
    a2.plot(run["t"], run["i_p"] * 1e3, "--", color=ORANGE)
    a2.set_ylabel("i_p (mA)", color=ORANGE); a2.grid(False)
    a.set_title(f"{r.Vac_V:.0f} V, {r.f_Hz:.0f} Hz, R_s = {r.Rload_ohm:.0f} Ω, B_max = {r.B_max_T:.2f} T", fontsize=9)
    a.set_xlabel("t / T"); a.set_ylabel("u_s (V)", color=BLUE)
ax[0].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.show()

# %% [markdown]
# The sign of the COMSOL currents follows its own convention; compare the shapes and amplitudes. In saturation
# the primary current becomes peaked: the core needs much more field for the last bit of flux. Now all runs at
# once – COMSOL against the magnetic circuit:

# %%
quant = [("V2_rms_V", "secondary voltage (V, rms)"), ("I1_rms_A", "primary current (A, rms)"),
         ("P_core_W", "core loss (W)"), ("P_load_W", "power in the load (W)")]


def parity(ax, y_true, y_pred, title, color=BLUE, label=None, log=True):
    ax.plot(y_true, y_pred, "o", ms=4, color=color, label=label)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(title, fontsize=9); ax.set_xlabel("COMSOL")


fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, (q, lab) in zip(ax, quant):
    parity(a, tr[q].values, circ[q].values, lab)
    err = np.abs(circ[q] / tr[q] - 1)
    a.text(0.05, 0.9, f"median error {np.median(err):.0%}", transform=a.transAxes, fontsize=8)
ax[0].set_ylabel("magnetic circuit")
plt.tight_layout(); plt.show()

# %% [markdown]
# A one-dimensional circuit with a guessed B(H) curve reproduces the 3D finite-element results with median errors
# of 2–7 % over four orders of magnitude – it captures the physics (Faraday, Ampère, saturation, even the peaked
# current of the saturated core), but not the leakage flux, the non-uniform flux density in the corners or the
# exact iron curve. The largest deviation is the core loss of the weakly magnetized runs (lower left, up to
# ~40 % too high): the circuit puts all the flux into the iron with a uniform B, while in 3D part of it leaks
# through the air and the loss $\propto B^{1.5}$ is distributed non-uniformly.

# %% [markdown]
# ### 19.2 The permanent-magnet generator
#
# The *Generator in 2D* of the COMSOL Application Library: 8 Nd-Fe-B magnets ($B_r = 1.41$ T) on a soft-iron rotor,
# a slotted stator with 8 coils of 100 turns, length 0.4 m, 60 rpm (4 Hz electrical). COMSOL solves the 2D
# transient problem with a rotating rotor; eddy currents in the (conducting) magnets are included, the laminated
# iron is non-conducting and its loss is computed afterwards with the Bertotti formula. The dataset varies the iron:
# its permeability (the B(H) curve is scaled, $B_{new}(H) = B(k_\mu H)$, which changes the initial permeability by
# $k_\mu$ but keeps the saturation; `mur_stator_mean` is the resulting mean $\mu_r$ of the stator), its conductivity
# $\sigma$ and the lamination thickness $d$.

# %%
gen = pd.read_csv(fetch_data("generator_2d", "generator_losses.csv"))
gen_emf = pd.read_csv(fetch_data("generator_2d", "generator_emf.csv"))
ref = gen[(gen.sigma_S_per_m == 1e7) & (gen.d_mm == 0.35)].sort_values("kmu")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].semilogx(ref.mur_stator_mean, ref.emf_peak_V, "o-", color=BLUE)
ax[0].set_xlabel("mean μr of the stator iron"); ax[0].set_ylabel("EMF amplitude (V)")
for col, lab, c in [("P_hyst_W", "hysteresis", BLUE), ("P_excess_W", "excess", ORANGE),
                    ("P_classical_W", "classical eddy", "gray"), ("P_magnets_W", "magnets (eddy)", "green")]:
    ax[1].semilogx(ref.mur_stator_mean, ref[col], "o-", ms=3, color=c, label=lab)
ax[1].set_xlabel("mean μr of the stator iron"); ax[1].set_ylabel("loss (W)"); ax[1].legend(fontsize=8)
ax[1].set_title("σ = 10 MS/m, d = 0.35 mm", fontsize=9)
k1 = gen[(gen.kmu == 1.0) & (gen.sigma_S_per_m > 0)]
for d, c in zip(sorted(k1.d_mm.unique()), plt.cm.Blues(np.linspace(0.35, 1, 5))):
    s = k1[k1.d_mm == d].sort_values("sigma_S_per_m")
    ax[2].plot(s.sigma_S_per_m / 1e6, s.P_iron_W, "o-", ms=3, color=c, label=f"d = {d} mm")
ax[2].set_xlabel("σ (MS/m)"); ax[2].set_ylabel("iron loss (W)"); ax[2].legend(fontsize=7); ax[2].set_title("original iron", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# The EMF saturates: above $\mu_r \approx 500$ the iron is practically ideal and the magnetic circuit is dominated by
# the magnets and the air gap (the answer to the prediction question). The hysteresis loss follows the flux density
# in the stator. At the low frequency of this machine (4 Hz) the classical eddy loss is tiny; the conductivity acts
# mostly through the excess term ($\propto\sqrt\sigma$).
#
# **Second method: our own 2D finite-element solver.** The magnetostatic problem $-\nabla\cdot(\nu\nabla A_z) =
# \nabla\times(\nu\mathbf B_r)$ is solved on a polar mesh of one pole pitch (45°, anti-periodic boundaries) with
# linear triangles and Newton's method for the nonlinear iron; the rotor is turned step by step (no eddy currents:
# every rotor position is a static problem). The geometry was rebuilt from the documentation of the COMSOL model,
# the iron curve is the analytic one of the transformer, scaled by $k_\mu$ in the same way. The EMF follows from
# the flux linkage $\psi(t)$ of the coils, the losses from the Bertotti formula applied to $B(t)$ in every element.

# %%
AIR, IRON, MAGNET = 0, 1, 2


class FEMGenerator:
    """2D magnetostatic finite-element model of the PM generator (one pole pitch, rotating rotor)."""

    def __init__(self, n_theta=45, kmu=1.0, poles=8, length=0.4, r_shaft=0.05, r_rotor=0.15, magnet_h=0.065,
                 magnet_w=0.10, r_tip=0.235, shoe_h=0.0175, shoe_w=0.115, tooth_w=0.09, r_slot=0.34, r_out=0.40,
                 Br=1.41, mu_rec=1.05, mu_i=1200.0, B0=1.25, p=8.0, turns=100, rpm=60.0, coil_w=0.04,
                 coil_r=(0.27, 0.33)):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.mu_i = mu_i * kmu                                   # B_new(H) = B(kmu H)  <=>  mu_i -> kmu mu_i
        self.pitch = 2 * np.pi / poles
        self.nt, self.dth = n_theta, self.pitch / n_theta
        r_mag = r_rotor + magnet_h
        r_mid = 0.5 * (r_mag + r_tip)                            # sliding surface in the air gap
        r = [r_shaft]
        for a, b, n in [(r_shaft, r_rotor, 12), (r_rotor, r_mag, 16), (r_mag, r_mid, 4), (r_mid, r_tip, 4),
                        (r_tip, r_tip + shoe_h, 5), (r_tip + shoe_h, r_slot, 16), (r_slot, r_out, 10)]:
            r += list(np.linspace(a, b, n + 1)[1:])
        self.r = np.array(r); self.nr = len(r)
        nr, nt = self.nr, self.nt
        ii, jj = [x.ravel() for x in np.meshgrid(np.arange(nr - 1), np.arange(nt), indexing="ij")]
        corners = [((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))]   # two triangles per polar cell
        vi = np.concatenate([np.stack([ii + c[k][0] for k in range(3)], 1) for c in corners])
        vj = np.concatenate([np.stack([jj + c[k][1] for k in range(3)], 1) for c in corners])
        x, y = self.r[vi] * np.cos(vj * self.dth), self.r[vi] * np.sin(vj * self.dth)
        self.sign = np.where(vj == nt, -1.0, 1.0)                # anti-periodic wrap at 45 degrees
        self.node = vi * nt + (vj % nt)
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
        area2 = (x * b).sum(1)
        self.area, self.gx, self.gy = 0.5 * area2, b / area2[:, None], c / area2[:, None]
        xc, yc = x.mean(1), y.mean(1)
        self.rc, self.thc = np.hypot(xc, yc), np.arctan2(yc, xc)
        fixed = np.zeros(nr * nt, bool); fixed[:nt] = fixed[-nt:] = True   # A = 0 at the shaft and outside
        self.free, self.A = np.flatnonzero(~fixed), np.zeros(nr * nt)
        ys = self.rc * np.sin(self.thc - 0.5 * self.pitch)       # the tooth is centred in the sector
        shoe = (self.rc > r_tip) & (self.rc < r_tip + shoe_h) & (np.abs(ys) < 0.5 * shoe_w)
        tooth = (self.rc > r_tip + shoe_h) & (self.rc < r_slot) & (np.abs(ys) < 0.5 * tooth_w)
        self.stator = np.where(shoe | tooth | (self.rc > r_slot), IRON, AIR)
        side = (np.abs(ys) > 0.5 * tooth_w) & (np.abs(ys) < 0.5 * tooth_w + coil_w) & \
            (self.rc > coil_r[0]) & (self.rc < coil_r[1])
        self.coil_plus, self.coil_minus = side & (ys > 0), side & (ys < 0)

    def materials(self, step):
        """Materials and remanence for the rotor turned by `step` angular divisions."""
        alpha = step * self.dth
        k = np.round((self.thc - alpha - 0.5 * self.pitch) / self.pitch)
        axis = alpha + 0.5 * self.pitch + k * self.pitch          # axis of the nearest magnet
        xm, ym = self.rc * np.cos(self.thc - axis), self.rc * np.sin(self.thc - axis)
        mat = self.stator.copy(); mat[self.rc < self.r_rotor] = IRON
        magnet = (self.rc > self.r_rotor) & (xm < self.r_rotor + self.magnet_h) & (np.abs(ym) < 0.5 * self.magnet_w) \
            & (self.rc < 0.5 * (self.r_rotor + self.magnet_h + self.r_tip))
        mat[magnet] = MAGNET
        pol = np.where(k % 2 == 0, 1.0, -1.0) * self.Br
        return mat, np.where(magnet, pol * np.cos(axis), 0.0), np.where(magnet, pol * np.sin(axis), 0.0)

    def solve(self, step, tol=1e-8):
        """Newton iteration for the nonlinear magnetostatic problem; returns B in every element."""
        mat, brx, bry = self.materials(step)
        s, nd, nn = self.sign, self.node, self.nr * self.nt
        nu_m, nu0 = 1 / (MU0 * self.mu_rec), 1 / (MU0 * self.mu_i)
        rhs_m = self.area[:, None] * nu_m * (brx[:, None] * self.gy - bry[:, None] * self.gx) * s
        A = self.A
        for it in range(50):
            a = A[nd] * s
            gx, gy = (a * self.gx).sum(1), (a * self.gy).sum(1)   # B = (gy, -gx)
            b2 = gx ** 2 + gy ** 2
            nu, dnu = np.full(len(b2), 1 / MU0), np.zeros(len(b2))
            nu[mat == MAGNET] = nu_m
            q = (b2[mat == IRON] / self.B0 ** 2) ** (self.p / 2)     # iron: nu = nu0 (1 + (B/B0)^p)
            nu[mat == IRON] = nu0 * (1 + q)
            dnu[mat == IRON] = nu0 * self.p / 2 * q / np.maximum(b2[mat == IRON], 1e-30)
            gN = gx[:, None] * self.gx + gy[:, None] * self.gy
            res = np.zeros(nn)
            np.add.at(res, nd.ravel(), ((self.area * nu)[:, None] * gN * s - rhs_m).ravel())
            K = (self.area * nu)[:, None, None] * (self.gx[:, :, None] * self.gx[:, None, :] +
                                                   self.gy[:, :, None] * self.gy[:, None, :]) \
                + (2 * self.area * dnu)[:, None, None] * gN[:, :, None] * gN[:, None, :]
            K = K * s[:, :, None] * s[:, None, :]
            J = sp.coo_matrix((K.ravel(), (np.repeat(nd, 3, 1).ravel(), np.tile(nd, (1, 3)).ravel())), (nn, nn)).tocsr()
            dA = spla.spsolve(J[self.free][:, self.free].tocsc(), -res[self.free])
            A = A.copy(); A[self.free] += dA
            if np.abs(dA).max() < tol * max(np.abs(A).max(), 1e-12):
                break
        self.A = A
        a = A[nd] * s
        return (a * self.gy).sum(1), -(a * self.gx).sum(1)

    def rotate(self):
        """One electrical period (two pole pitches): EMF and B(t) in the stator and rotor frame."""
        n, omega = 2 * self.nt, 2 * np.pi * self.rpm / 60
        psi, Br, Bt = np.zeros(n), np.zeros((n, len(self.area))), np.zeros((n, len(self.area)))
        c, sn = np.cos(self.thc), np.sin(self.thc)
        for k in range(n):
            Bx, By = self.solve(k)
            ac = (self.A[self.node] * self.sign).mean(1)
            mean = lambda m: (ac[m] * self.area[m]).sum() / self.area[m].sum()
            psi[k] = self.poles * self.turns * self.length * (mean(self.coil_plus) - mean(self.coil_minus))
            Br[k], Bt[k] = Bx * c + By * sn, -Bx * sn + By * c
        shape, j = (2, self.nr - 1, self.nt), np.arange(self.nt)
        Br_rot, Bt_rot = np.zeros((n,) + shape), np.zeros((n,) + shape)
        for k in range(n):                                       # follow a point of the rotor
            jj, sg = (j + k) % self.nt, np.where(((j + k) // self.nt) % 2 == 0, 1.0, -1.0)
            Br_rot[k], Bt_rot[k] = Br[k].reshape(shape)[:, :, jj] * sg, Bt[k].reshape(shape)[:, :, jj] * sg
        emf = (np.roll(psi, -1) - np.roll(psi, 1)) / (2 * self.dth / omega)   # coil voltage d(psi)/dt, as COMSOL
        return dict(t=np.arange(n) * self.dth / omega, emf=emf, Br=Br, Bt=Bt,
                    Br_rot=Br_rot.reshape(n, -1), Bt_rot=Bt_rot.reshape(n, -1))

    def iron_loss(self, run, sigma=1e7, d=0.35e-3, kh=1000.0, beta=1.5, S=1e-3, V0=0.5):
        """Bertotti loss (W) of the stator and rotor iron, harmonic by harmonic."""
        f1 = self.rpm / 60 * self.poles / 2
        rotor = (self.materials(0)[0] == IRON) & (self.rc < self.r_rotor)
        out = dict(hysteresis=0.0, classical=0.0, excess=0.0)
        for mask, Br, Bt in ((self.stator == IRON, run["Br"], run["Bt"]), (rotor, run["Br_rot"], run["Bt_rot"])):
            n = Br.shape[0]
            amp = 2 / n * np.sqrt(np.abs(np.fft.rfft(Br[:, mask], axis=0)) ** 2 +
                                  np.abs(np.fft.rfft(Bt[:, mask], axis=0)) ** 2)[1:n // 2]
            f = f1 * np.arange(1, n // 2)[:, None]
            dv = self.area[mask] * self.length * self.poles
            out["hysteresis"] += (kh * f * amp ** beta).sum(0) @ dv
            out["classical"] += ((np.pi * d) ** 2 / 6 * sigma * f ** 2 * amp ** 2).sum(0) @ dv
            out["excess"] += (8 * np.sqrt(sigma * 0.1356 * S * V0) * (f * amp) ** 1.5).sum(0) @ dv
        return out


t0 = time.time()
fem = {}
for kmu in [0.01, 0.03323, 0.1104, 0.3669, 1.0, 3.0]:              # values of the COMSOL dataset
    g = FEMGenerator(kmu=kmu)
    run = g.rotate()
    fem[kmu] = dict(run=run, loss=g.iron_loss(run), emf=np.abs(run["emf"]).max())
    print(f"kmu = {kmu:6.4g}: EMF amplitude {fem[kmu]['emf']:5.0f} V, iron loss "
          + ", ".join(f"{k} {v:5.1f} W" for k, v in fem[kmu]["loss"].items()))
print(f"computed in {time.time() - t0:.0f} s")

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for kmu, c in [(1.0, BLUE), (0.01, ORANGE)]:
    e = gen_emf[np.isclose(gen_emf.kmu, kmu)]
    ax[0].plot(e.t_s, e.emf_V, color=c, label=f"COMSOL, kmu = {kmu}")
    ax[0].plot(fem[kmu]["run"]["t"], fem[kmu]["run"]["emf"], "--", color=c, label=f"own FEM, kmu = {kmu}")
ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("EMF (V)"); ax[0].legend(fontsize=7)
ks = np.array(sorted(fem))
cm = ref.set_index("kmu").loc[ks]
ax[1].semilogx(ks, cm.emf_peak_V, "o-", color=BLUE, label="COMSOL")
ax[1].semilogx(ks, [fem[k]["emf"] for k in ks], "s--", color=ORANGE, label="own FEM")
ax[1].set_xlabel("permeability factor kmu"); ax[1].set_ylabel("EMF amplitude (V)"); ax[1].legend()
for col, key, c in [("P_hyst_W", "hysteresis", BLUE), ("P_excess_W", "excess", ORANGE),
                    ("P_classical_W", "classical", "gray")]:
    ax[2].semilogx(ks, cm[col], "o-", color=c, label=f"{key}, COMSOL")
    ax[2].semilogx(ks, [fem[k]["loss"][key] for k in ks], "s--", color=c, label=f"{key}, own FEM")
ax[2].set_xlabel("permeability factor kmu"); ax[2].set_ylabel("iron loss (W), σ = 10 MS/m, d = 0.35 mm")
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
for k in ks:
    print(f"kmu = {k:6.4g}: EMF  COMSOL {cm.emf_peak_V[k]:5.0f} V, own FEM {fem[k]['emf']:5.0f} V | iron loss  "
          f"COMSOL {cm.P_iron_W[k]:5.0f} W, own FEM {sum(fem[k]['loss'].values()):5.0f} W")

# %% [markdown]
# The two finite-element codes agree on the EMF to about one per cent over the whole permeability range – the
# field problem is well posed and both solvers converge to the same answer. The iron losses agree in their trends
# but our values are some 8–16 % higher. The loss is a *post-processing* quantity that depends on the local
# flux density to the power 1.5–2, so it is far more sensitive to the iron curve near the knee (we use an analytic
# approximation), to the mesh in the tooth tips and to how the harmonics are summed. This is the general lesson:
# integral quantities (EMF, flux) are robust, local and nonlinear quantities (losses, peak fields) much less so.

# %% [markdown]
# ### 19.3 Neural-network surrogates
#
# A COMSOL run takes seconds (generator) to minutes (transformer); a design study or an optimization needs
# thousands. A **surrogate** – a small neural network trained on the simulation results – answers in microseconds.
# We train one surrogate per device with the same recipe as in the previous case studies: a fully connected network
# (SiLU activations), inputs and outputs standardized, positive outputs learned as logarithms, full-batch Adam with
# a cosine learning-rate schedule, the best state on a validation set is kept.

# %%
class Surrogate(torch.nn.Module):
    def __init__(self, x, y, width=64, depth=3):
        super().__init__()
        self.register_buffer("xm", torch.tensor(x.mean(0))); self.register_buffer("xs", torch.tensor(x.std(0) + 1e-9))
        self.register_buffer("ym", torch.tensor(y.mean(0))); self.register_buffer("ys", torch.tensor(y.std(0) + 1e-9))
        layers, n = [], x.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]; n = width
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(n, y.shape[1]))
        self.float()                    # magnum.np (sections 11–15) switches the torch default to float64

    def forward(self, x):
        return self.net((x - self.xm) / self.xs) * self.ys + self.ym

    def predict(self, x):
        with torch.no_grad():
            return self(torch.tensor(np.asarray(x, np.float32))).numpy()


def train_surrogate(x_tr, y_tr, x_va, y_va, width=64, depth=3, epochs=4000, lr=3e-3, wd=0.0, seed=0, verbose=True):
    torch.manual_seed(seed)
    x_tr, y_tr, x_va, y_va = [torch.tensor(np.asarray(a, np.float32)) for a in (x_tr, y_tr, x_va, y_va)]
    model = Surrogate(x_tr.numpy(), y_tr.numpy(), width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scale = model.ys
    best, best_state, hist = np.inf, None, []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.mean(((model(x_tr) - y_tr) / scale) ** 2)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            val = torch.mean(((model(x_va) - y_va) / scale) ** 2).item()
        hist.append((loss.item(), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if verbose:
        print(f"best validation loss {best:.2e} (standardized units), final training loss {hist[-1][0]:.2e}")
    return model, np.array(hist)


def rel_err(pred, true):
    return np.abs(np.asarray(pred) / np.asarray(true) - 1)


# %% [markdown]
# **Generator surrogate.** Inputs: $\log_{10}k_\mu$, $\sigma$ (MS/m) and $d$ (mm); outputs: EMF amplitude, total iron
# loss and loss in the magnets. The 756 COMSOL rows lie on a grid of 21 permeabilities × 36 (σ, d) combinations.
# To test **interpolation** we hold out four complete permeability values (all their σ, d) as the test set, and
# 15 % of the rest for validation.

# %%
def gen_x(df):
    return np.column_stack([np.log10(df.kmu), df.sigma_S_per_m / 1e6, df.d_mm])


def gen_y(df):
    return np.column_stack([df.emf_peak_V / 100, np.log(df.P_iron_W), np.log(df.P_magnets_W)])


def gen_y_inv(y):
    return np.column_stack([y[:, 0] * 100, np.exp(y[:, 1]), np.exp(y[:, 2])])


rng = np.random.default_rng(1)
test_k = [0.0135, 0.1104, 0.6687, 1.646]
g_test = gen[gen.kmu.isin(test_k)]
g_rest = gen[~gen.kmu.isin(test_k)]
va = rng.random(len(g_rest)) < 0.15
g_tr, g_va = g_rest[~va], g_rest[va]
print(f"generator: {len(g_tr)} training, {len(g_va)} validation, {len(g_test)} test rows")
t0 = time.time()
gen_model, gen_hist = train_surrogate(gen_x(g_tr), gen_y(g_tr), gen_x(g_va), gen_y(g_va))
print(f"training time {time.time() - t0:.0f} s")
g_pred = gen_y_inv(gen_model.predict(gen_x(g_test)))
g_cols = [("emf_peak_V", "EMF amplitude (V)"), ("P_iron_W", "iron loss (W)"), ("P_magnets_W", "magnet loss (W)")]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
ax[0].semilogy(gen_hist[:, 0], color=BLUE, label="training"); ax[0].semilogy(gen_hist[:, 1], color=ORANGE, label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
for a, (col, lab), k in zip(ax[1:], g_cols, range(3)):
    parity(a, g_test[col].values, g_pred[:, k], lab, log=False)
    e = rel_err(g_pred[:, k], g_test[col].values)
    a.text(0.05, 0.85, f"median {np.median(e):.2%}\nmax {e.max():.2%}", transform=a.transAxes, fontsize=8)
ax[1].set_ylabel("DNN (test rows)")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Extrapolation.** The same network, now trained only on $k_\mu \ge 0.033$ and asked about the three least
# permeable irons ($k_\mu$ = 0.010–0.025, where the EMF drops fastest):

# %%
low = gen.kmu < 0.03
g_rest2 = gen[~low]
va2 = rng.random(len(g_rest2)) < 0.15
ext_model, _ = train_surrogate(gen_x(g_rest2[~va2]), gen_y(g_rest2[~va2]), gen_x(g_rest2[va2]), gen_y(g_rest2[va2]),
                               verbose=False)
kk = np.logspace(-2.2, 0.6, 80)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, k, lab in [(ax[0], 0, "EMF amplitude (V)"), (ax[1], 1, "iron loss (W)")]:
    xq = np.column_stack([np.log10(kk), np.full_like(kk, 10.0), np.full_like(kk, 0.35)])
    sel = ref
    a.semilogx(sel.kmu, sel[g_cols[k][0]], "o", color="k", label="COMSOL")
    a.semilogx(kk, gen_y_inv(gen_model.predict(xq))[:, k], color=BLUE, label="DNN, trained on all kmu (except test)")
    a.semilogx(kk, gen_y_inv(ext_model.predict(xq))[:, k], "--", color=ORANGE, label="DNN, trained on kmu ≥ 0.033")
    a.axvspan(kk[0], 0.03, color="gray", alpha=0.2)
    a.set_xlabel("permeability factor kmu (σ = 10 MS/m, d = 0.35 mm)"); a.set_ylabel(lab)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
e_ext = rel_err(gen_y_inv(ext_model.predict(gen_x(gen[low])))[:, 0], gen[low].emf_peak_V)
print(f"EMF error outside the training range: median {np.median(e_ext):.1%}, max {e_ext.max():.1%}")

# %% [markdown]
# Inside the range of its training data the surrogate is excellent (errors of a few hundredths of a per cent);
# outside it (grey band) the error jumps by two orders of magnitude and keeps growing with the distance from the
# data – the network simply continues the trend in some smooth way. A surrogate is an *interpolator*: its validity
# ends where the data end, and nothing in the network warns you when you leave that range.

# %% [markdown]
# **Transformer surrogate.** Inputs: $\log_{10}V_{ac}$, $\log_{10}f$, $\log_{10}R_s$; outputs: logarithms of the
# secondary voltage, primary current, core loss and load power. Here we have only a few dozen expensive runs, so we
# compare three predictors on the same held-out test runs:
# * the **DNN** trained on the COMSOL runs alone;
# * the **magnetic circuit** (physics only, no training);
# * a **hybrid**: the circuit result corrected by a small network that learns only the ratio COMSOL / circuit
#   (*residual learning* – the physics carries the large variations over orders of magnitude, the network the
#   smaller, smooth correction).

# %%
tq = [q for q, _ in quant]
x_t = np.log10(tr[["Vac_V", "f_Hz", "Rload_ohm"]].values)
y_t = np.log(tr[tq].values)
y_c = np.log(circ[tq].values)
rng = np.random.default_rng(3)
idx = rng.permutation(len(tr))
n_te, n_va = max(6, len(tr) // 5), max(4, len(tr) // 8)
te, va, trn = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
print(f"transformer: {len(trn)} training, {len(va)} validation, {len(te)} test runs")
dnn_t, hist_t = train_surrogate(x_t[trn], y_t[trn], x_t[va], y_t[va], width=32, depth=2, epochs=4000, wd=1e-4)
hyb_t, hist_h = train_surrogate(x_t[trn], (y_t - y_c)[trn], x_t[va], (y_t - y_c)[va], width=16, depth=2,
                                epochs=3000, wd=1e-3)
pred = {"DNN": np.exp(dnn_t.predict(x_t[te])), "magnetic circuit": np.exp(y_c[te]),
        "hybrid (circuit + DNN)": np.exp(y_c[te] + hyb_t.predict(x_t[te]))}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for k, (a, (q, lab)) in enumerate(zip(ax, quant)):
    for (name, p), c in zip(pred.items(), [BLUE, "gray", ORANGE]):
        parity(a, tr[q].values[te], p[:, k], lab, color=c, label=name)
ax[0].set_ylabel("prediction (test runs)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
print("median / max relative error on the test runs")
print(f"{'':24s}" + "".join(f"{q:>20s}" for q in tq))
for name, p in pred.items():
    e = rel_err(p, tr[tq].values[te])
    print(f"{name:24s}" + "".join(f"{np.median(e[:, k]):>11.1%} /{e[:, k].max():>6.0%}" for k in range(len(tq))))

# %% [markdown]
# ### 19.4 How much data does the surrogate need?
#
# The test error of the transformer DNN as a function of the number of training runs (the test runs stay the same;
# for each size the network is retrained on a random subset). The horizontal lines are the magnetic circuit
# (no training at all).

# %%
sizes = [n for n in (8, 12, 16, 24, len(trn)) if n <= len(trn)]
curves = {"DNN": [], "hybrid (circuit + DNN)": []}
t0 = time.time()
for n in sizes:
    e_d, e_h = [], []
    for seed in range(3):
        sub = np.random.default_rng(seed).choice(trn, n, replace=False)
        m1, _ = train_surrogate(x_t[sub], y_t[sub], x_t[va], y_t[va], width=32, depth=2, epochs=2000, wd=1e-4,
                                seed=seed, verbose=False)
        m2, _ = train_surrogate(x_t[sub], (y_t - y_c)[sub], x_t[va], (y_t - y_c)[va], width=16, depth=2,
                                epochs=2000, wd=1e-3, seed=seed, verbose=False)
        e_d.append(np.median(rel_err(np.exp(m1.predict(x_t[te])), tr[tq].values[te])))
        e_h.append(np.median(rel_err(np.exp(y_c[te] + m2.predict(x_t[te])), tr[tq].values[te])))
    curves["DNN"].append(np.mean(e_d)); curves["hybrid (circuit + DNN)"].append(np.mean(e_h))
plt.figure(figsize=(6.5, 4))
plt.loglog(sizes, curves["DNN"], "o-", color=BLUE, label="DNN")
plt.loglog(sizes, curves["hybrid (circuit + DNN)"], "s-", color=ORANGE, label="hybrid (circuit + DNN)")
plt.axhline(np.median(rel_err(np.exp(y_c[te]), tr[tq].values[te])), color="gray", ls="--", label="magnetic circuit")
plt.xlabel("number of COMSOL training runs"); plt.ylabel("median relative error (test runs, all outputs)")
ax_lc = plt.gca()
ax_lc.set_xticks(sizes, [str(n) for n in sizes]); ax_lc.xaxis.set_minor_formatter(plt.NullFormatter())
ax_lc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_lc.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if round(v * 100) in (2, 3, 5) else ""))
plt.legend(); plt.show()
print(f"computed in {time.time() - t0:.0f} s")

# %% [markdown]
# ### 19.5 Discussion
#
# * **From the domain wall to the device.** The device models contain no micromagnetics – the material enters
#   through a B(H) curve and three loss constants. Sections 3–17 explain what these numbers mean: the B(H) curve is the
#   average response of many domains, the hysteresis constant $k_h$ is the loop area set by wall pinning, and the
#   excess loss comes from eddy currents around moving walls. Micromagnetics explains *why* and *how to improve*
#   a material; the measured curves are what a device designer uses.
# * **Two simulation methods.** For the generator, our compact 2D finite-element code reproduces the COMSOL EMF to
#   about 1 % – a strong check of both. The losses differ by 8–16 %, because they are local, nonlinear
#   post-processed quantities. For the transformer, the 1D magnetic circuit captures the physics (Faraday, Ampère,
#   saturation) with a fraction of the effort, but misses leakage and the 3D distribution of the flux. Whenever
#   two independent methods agree, we trust the result; where they disagree, we learn where the modelling
#   assumptions matter.
# * **The surrogates.** With hundreds of cheap runs (generator) a DNN interpolates to a fraction of a per cent – far
#   below the difference between the two simulation methods, i.e. the surrogate is not the weakest link. With a few
#   dozen expensive runs (transformer) the pure DNN is limited by the data: trained on 33 runs it is about as good
#   as the magnetic circuit in the median (~4–5 %), but with outliers of tens of per cent. Combining it with the
#   physics model (*hybrid* or *residual* learning) uses the data much more efficiently: with only 8 runs the hybrid
#   is already twice as accurate as both, with all runs it reaches 1–3 %. In both cases the surrogate fails
#   outside the range of its training data, and it inherits every modelling error of the simulation it was
#   trained on – a DNN trained on COMSOL is never better than COMSOL.
# * **What decides the losses?** Above $\mu_r \approx 500$ the permeability barely matters for the generator: the
#   magnets and the air gap dominate. The hysteresis loss dominates at 4 Hz; at 50 Hz and above the eddy-current
#   terms (∝ $\sigma d^2 f^2$ and $\sqrt\sigma f^{1.5}$) take over – which is why transformer cores are thin,
#   silicon-alloyed (low σ) laminations, and why the coercivity of section 15 matters most at low frequency.

# %% [markdown]
# ## 20 Permanent magnets
#
# A permanent magnet works in the **second quadrant** of its loop: its own demagnetizing field drives it there.
# For a modern magnet the demagnetization curve is almost a straight line,
# $B = B_r + \mu_0\mu_{rec}H$ (recoil permeability $\mu_{rec}\approx 1.05$). The **energy product** $BH$ measures
# the field energy that the magnet can create in the air gap; its maximum is
# $$(BH)_{max} = \frac{B_r^2}{4\mu_0\mu_{rec}}\quad\text{at}\quad B = B_r/2 .$$
# The working point lies on the **load line** $B = -\mu_0\frac{1-N}{N}H$ of the magnet's shape (demagnetizing
# factor $N$, no iron): a thin disc magnetized through its thickness ($N\to1$) works at low $B$, a long rod at
# high $B$.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for name, Br_, mur_, c in [("NdFeB, Br = 1.3 T", 1.3, 1.05, BLUE), ("Sr ferrite, Br = 0.4 T", 0.4, 1.10, ORANGE)]:
    HcB = Br_ / (MU0 * mur_)
    H = np.linspace(-HcB, 0, 200); B = Br_ + MU0 * mur_ * H
    ax[0].plot(H / 1e3, B, color=c, label=name)
    ax[1].plot(-H * B / 1e3, B, color=c, label=f"{name}: (BH)max = {Br_**2 / (4 * MU0 * mur_) / 1e3:.0f} kJ/m³")
for Nd in (0.2, 0.5, 0.8):
    H = np.linspace(-1.2e6, 0, 10)
    ax[0].plot(H / 1e3, -MU0 * (1 - Nd) / Nd * H, ":", color=GREY)
    Hl = max(-1.15e6, -1.25 / (MU0 * (1 - Nd) / Nd))
    ax[0].text(Hl / 1e3 + 15, -MU0 * (1 - Nd) / Nd * Hl + 0.03, f"load line N = {Nd}", fontsize=8, color=GREY)
ax[0].set_xlim(-1300, 0); ax[0].set_ylim(0, 1.45)
ax[0].set_xlabel("H (kA/m)"); ax[0].set_ylabel("B (T)"); ax[0].legend(loc="lower left", fontsize=8)
ax[0].set_title("demagnetization curves and load lines")
ax[1].set_xlabel("energy product |B·H| (kJ/m³)"); ax[1].set_ylabel("B (T)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# The working point of a magnet is where its load line crosses its demagnetization curve. A magnet with
# $N \approx 0.5$ (e.g. a cube-like block) works close to $(BH)_{max}$ – this is how magnets are sized.
# Hard ferrites have ten times less $(BH)_{max}$, but cost a small fraction of NdFeB per kilogram and contain no
# rare earths – they are still the magnets produced in the largest mass.

# %% [markdown]
# ## 21 Soft and hard magnetic materials in numbers
#
# Typical room-temperature values (ranges of commercial grades; data sheets of the producers and the
# textbooks listed at the end).

# %%
soft = {  # name: (mu_max range, Bs range (T), Hc range (A/m))
    "MnZn ferrite": ((1e3, 1.5e4), (0.35, 0.5), (5, 20)),
    "NiZn ferrite": ((10, 1e3), (0.25, 0.4), (20, 300)),
    "Fe-Si non-oriented": ((3e3, 1.5e4), (1.9, 2.05), (30, 80)),
    "Fe-Si grain-oriented": ((2e4, 8e4), (1.95, 2.05), (5, 15)),
    "Fe-Co (Permendur)": ((5e3, 2e4), (2.2, 2.4), (15, 400)),
    "Ni-Fe 50 %": ((2e4, 1e5), (1.4, 1.6), (5, 25)),
    "Ni-Fe 80 % (permalloy)": ((1e5, 1e6), (0.7, 0.8), (0.3, 2)),
    "Fe-based amorphous": ((1e5, 6e5), (1.45, 1.6), (2, 5)),
    "Co-based amorphous": ((1e5, 1e6), (0.5, 0.8), (0.2, 1)),
    "nanocrystalline": ((3e4, 3e5), (1.15, 1.25), (0.5, 2)),
    "soft magnetic composite": ((200, 800), (1.4, 1.9), (150, 400)),
    "pure iron": ((5e3, 2e5), (2.1, 2.16), (10, 80)),
}
hard = {  # name: (Br range (T), HcJ range (kA/m), (BH)max range (kJ/m^3))
    "hard ferrite (Ba, Sr)": ((0.38, 0.45), (180, 400), (26, 40)),
    "Alnico": ((0.6, 1.35), (40, 160), (30, 80)),
    "SmCo₅": ((0.85, 1.0), (1200, 2400), (140, 200)),
    "Sm₂Co₁₇": ((1.0, 1.15), (800, 2400), (190, 260)),
    "NdFeB sintered": ((1.15, 1.5), (900, 2800), (250, 440)),
    "NdFeB bonded": ((0.6, 0.75), (600, 1000), (60, 100)),
}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
cols = plt.cm.tab20(np.linspace(0, 1, 20))
for k, (n, ((m1, m2), (b1, b2), _)) in enumerate(soft.items()):
    ax[0].fill_between([m1, m2], [b1, b1], [b2, b2], color=cols[k], alpha=0.6)
    below = n in ("Fe-Si grain-oriented", "Co-based amorphous", "Fe-Si non-oriented")
    ax[0].text(np.sqrt(m1 * m2), b1 - 0.1 if below else b2 + 0.03, n, ha="center", fontsize=7.5)
ax[0].set_xscale("log"); ax[0].set_xlim(5, 3e6); ax[0].set_ylim(0, 2.6)
ax[0].set_xlabel("maximum relative permeability µ_max"); ax[0].set_ylabel("saturation B_s (T)")
ax[0].set_title("soft magnetic materials")
for k, (n, ((b1, b2), (h1, h2), _)) in enumerate(hard.items()):
    ax[1].fill_between([h1, h2], [b1, b1], [b2, b2], color=cols[2 * k], alpha=0.6)
    ax[1].text(np.sqrt(h1 * h2), b2 + 0.03, n, ha="center", fontsize=8)
for BH in (25, 100, 400):
    Br_line = np.sqrt(4 * MU0 * 1.05 * BH * 1e3)
    ax[1].axhline(Br_line, color=GREY, ls=":", lw=0.8)
    ax[1].text(25, Br_line + 0.015, f"(BH)max = {BH} kJ/m³ for a linear magnet", fontsize=7, color=GREY)
ax[1].set_xscale("log"); ax[1].set_xlim(20, 5000); ax[1].set_ylim(0, 1.7)
ax[1].set_xlabel("intrinsic coercivity H_cJ (kA/m)"); ax[1].set_ylabel("remanence B_r (T)")
ax[1].set_title("permanent magnets")
plt.tight_layout(); plt.show()

# %% [markdown]
# The coercive fields of the soft (listed in the code, in A/m) and hard (right, in kA/m) materials
# differ by up to seven orders of magnitude, although all of them are built from Fe, Co and Ni atoms with similar
# moments. The difference is made by anisotropy and **microstructure**: soft materials are designed so that domain
# walls move easily (low anisotropy, few defects, large grains or – in amorphous and nanocrystalline alloys –
# averaged-out anisotropy), permanent magnets so that reversed domains can neither nucleate nor grow (high
# anisotropy, small grains isolated by grain-boundary phases).
#
# **A century of permanent magnets** – the maximum energy product of the best commercial magnets (approximate
# values):

# %%
hist = [(1917, 8, "KS steel"), (1931, 10, "MK steel / Alnico"), (1948, 40, "Alnico 5"),
        (1955, 28, "hard ferrite"), (1958, 60, "columnar Alnico"), (1967, 160, "SmCo₅"),
        (1977, 240, "Sm₂Co₁₇"), (1984, 290, "NdFeB"), (1995, 400, "NdFeB"), (2005, 440, "NdFeB")]
plt.figure(figsize=(8, 4))
yr, bh, lab = zip(*hist)
plt.semilogy(yr, bh, "o", color=BLUE)
for y, b, l in hist:
    plt.annotate(l, (y, b), textcoords="offset points", xytext=(4, -10 if l == "hard ferrite" else 4), fontsize=8)
plt.xlabel("year"); plt.ylabel("(BH)max (kJ/m³)"); plt.xlim(1910, 2020); plt.ylim(5, 800)
plt.title("doubling roughly every 14 years until 1990"); plt.show()

# %% [markdown]
# ## 22 Magnetic recording and spintronics
#
# **Predict:** a hard-disk bit is written into grains of a few nanometres. Why can the grains not simply be made
# ten times smaller? And how can a memory cell be switched by a current instead of a magnetic field?
#
# ### 22.1 Recording and the superparamagnetic limit
#
# A hard disk stores a bit in a group of ~10 grains of a hard magnetic film (perpendicular CoCrPt-oxide), each grain
# a single-domain particle (section 11) with its easy axis perpendicular to the disk. The write head – a small
# electromagnet with a FeCo pole – reverses the grains; the read head senses their stray field (section 22.2). Higher
# areal density needs **smaller grains**, but the energy barrier between "up" and "down" of a grain is
# $\Delta E = K_uV$ (Stoner–Wohlfarth, section 8), and the thermal energy tries to cross it. The mean time between
# thermally activated reversals is given by the **Néel–Arrhenius law**
# $$\tau = \tau_0\exp\!\left(\frac{K_uV}{k_BT}\right),\qquad \tau_0\approx 10^{-9}\ \text{s}.$$
# Storing data for 10 years needs $K_uV/k_BT \gtrsim 40$; practical designs use $\approx 60$. Below this size a
# grain is **superparamagnetic**: it is ferromagnetic inside, but its moment flips randomly.

# %%
kT = K_B * 300
h_grain = 10e-9                                     # grain height (film thickness)
D = np.linspace(1, 14, 300) * 1e-9                  # grain diameter
media = [("CoCrPt-oxide (today's perpendicular media)", 0.5e6, 0.5e6),   # (name, Ku in J/m^3, Ms in A/m)
         ("FePt L1₀ (heat-assisted recording)", 6.6e6, 1.1e6)]
year = 3.15e7
fig, ax = plt.subplots(figsize=(8, 4))
for (name, Ku, Msm), c in zip(media, (BLUE, ORANGE)):
    V = np.pi * D**2 / 4 * h_grain
    tau = 1e-9 * np.exp(np.minimum(Ku * V / kT, 700))
    ax.semilogy(D * 1e9, tau, color=c, label=name)
    D60 = np.sqrt(4 * 60 * kT / Ku / (np.pi * h_grain))
    print(f"{name:45s} K_u = {Ku / 1e6:.1f} MJ/m^3, mu0 H_K = {2 * Ku / Msm:4.1f} T, "
          f"smallest stable grain (K_uV/kT = 60): {D60 * 1e9:.1f} nm")
for tt, lab in [(1, "1 s"), (3600, "1 hour"), (10 * year, "10 years"), (1e9 * year, "10$^9$ years")]:
    ax.axhline(tt, color=GREY, lw=0.8, ls=":"); ax.text(13.9, tt * 1.5, lab, ha="right", fontsize=8, color=GREY)
ax.set_ylim(1e-9, 1e20); ax.set_xlabel("grain diameter (nm), height 10 nm")
ax.set_ylabel("mean time to thermal reversal τ (s)"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Néel–Arrhenius: τ = τ0 exp(K_uV/k_BT)"); plt.tight_layout(); plt.show()

# %% [markdown]
# Notice how steep the curves are: one nanometre in diameter changes the lifetime from seconds to centuries. This
# is the **recording trilemma**: small grains (signal-to-noise ratio) need a large $K_u$ (thermal stability), but a
# large $K_u$ means a large anisotropy field $H_K = 2K_u/\mu_0M_s$ that the write head cannot reach – the best pole
# materials (FeCo) give at most about 2.4 T. FePt grains of 3 nm would be stable, but their $\mu_0H_K \approx 12$ T
# is far beyond any head. **Heat-assisted magnetic recording (HAMR)** solves this by heating the bit with a laser
# close to the Curie temperature (section 6) for a nanosecond while writing; **microwave-assisted recording (MAMR)**
# helps the reversal with a GHz field from a small oscillator in the head – the precession of section 14.
#
# ### 22.2 Reading: giant and tunnel magnetoresistance
#
# Since 1997 read heads use **spin-dependent transport**. In a ferromagnetic metal the electrons with spin
# parallel and antiparallel to $\mathbf M$ scatter differently (resistances $r_\uparrow$ and $r_\downarrow$), and
# the two spin channels conduct in parallel (Mott's two-current model). In a stack ferromagnet / thin spacer /
# ferromagnet the resistance depends on the **relative orientation** of the two magnetizations:
# * **GMR** (giant magnetoresistance, Fert and Grünberg 1988, Nobel prize 2007), metallic spacer (Cu):
#   $$\mathrm{GMR} = \frac{R_{AP}-R_P}{R_P} = \frac{(r_\downarrow - r_\uparrow)^2}{4r_\uparrow r_\downarrow};$$
# * **TMR** (tunnel magnetoresistance), insulating spacer of 1–2 nm – a **magnetic tunnel junction (MTJ)**. With
#   the spin polarizations $P_{1,2}$ of the two electrodes, Jullière's model gives
#   $$\mathrm{TMR} = \frac{R_{AP}-R_P}{R_P} = \frac{2P_1P_2}{1-P_1P_2}.$$
#   With amorphous Al₂O₃ barriers this gives the measured 20–70 %. Crystalline **MgO** barriers filter the electron
#   states by their symmetry (coherent tunnelling) and reach several hundred per cent at room temperature – much more
#   than Jullière's formula with the measured polarizations predicts.
#
# One electrode (the **reference layer**) is pinned by an antiferromagnet; the other (the **free layer**) follows
# the field of the disk, or stores the bit in a memory.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ratio = np.linspace(1, 12, 200)                                   # r_down / r_up
ax[0].plot(ratio, 100 * (ratio - 1)**2 / (4 * ratio), color=BLUE)
ax[0].set_xlabel("spin asymmetry of scattering r$_↓$/r$_↑$"); ax[0].set_ylabel("GMR (%)")
ax[0].set_title("GMR, two-current model")
Pp = np.linspace(0, 0.9, 200)
ax[1].plot(Pp, 100 * 2 * Pp**2 / (1 - Pp**2), color=BLUE, label="Jullière, $P_1 = P_2 = P$")
ax[1].axhspan(20, 70, color=GREY, alpha=0.2); ax[1].text(0.02, 80, "Al₂O₃ barriers (measured)", fontsize=8)
ax[1].axhline(600, color=ORANGE, ls="--"); ax[1].text(0.02, 620, "CoFeB/MgO/CoFeB, room temperature", fontsize=8,
                                                      color=ORANGE)
ax[1].set_ylim(0, 800); ax[1].set_xlabel("spin polarization P"); ax[1].set_ylabel("TMR (%)")
ax[1].set_title("TMR of a magnetic tunnel junction"); ax[1].legend(loc="center left", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 22.3 MRAM: writing a bit with a spin-polarized current
#
# A **magnetic random-access memory (MRAM)** stores the bit in the free layer of an MTJ (low resistance P = "0",
# high resistance AP = "1"); it is non-volatile, fast, and endures practically unlimited write cycles. The first
# MRAMs were written by the magnetic field of current lines – a field that does not scale down. Today's
# **STT-MRAM** (embedded in microcontrollers and as a cache since ~2019) uses the **spin-transfer torque (STT)**
# predicted by Slonczewski and Berger in 1996: electrons passing the reference layer become spin-polarized along its
# magnetization $\mathbf p$; in the free layer they hand their transverse angular momentum over to $\mathbf M$. The
# LLG equation of section 14 gets one more term (written here as an effective field, $\gamma = \mu_0\gamma_e$):
# $$\frac{\mathrm d\mathbf m}{\mathrm dt} = -\frac{\gamma}{1+\alpha^2}\Bigl[\mathbf m\times\mathbf h
#   + \alpha\,\mathbf m\times(\mathbf m\times\mathbf h)\Bigr],\qquad
#   \mathbf h = \mathbf H_{eff} + a_J\,\mathbf m\times\mathbf p,\qquad a_J = \frac{\hbar\,\varepsilon\,J}{e\,\mu_0M_st},$$
# where $J$ is the current density, $t$ the thickness of the free layer and $\varepsilon$ the spin-torque
# efficiency ($\varepsilon = P/2$ in Slonczewski's model with a symmetric junction). The new term acts like a
# **negative damping**: it pumps energy into the precession around the easy axis. When it exceeds the natural
# damping, the precession grows until $\mathbf m$ flips. For a free layer with perpendicular anisotropy
# ($H_K = 2K_{eff}/\mu_0M_s$) the balance gives the **critical current density**
# $$J_{c0} = \frac{\alpha}{\varepsilon}\,\frac{e\,\mu_0M_s\,t}{\hbar}\,H_K .$$
# Low damping $\alpha$, high polarization and a thin free layer make the memory efficient – which is why the free
# layer is ~1 nm of CoFeB on MgO (low α, perpendicular interface anisotropy, high TMR).
#
# **Predict:** with twice the critical current, does the bit switch in a picosecond, a nanosecond or a microsecond?

# %%
GAMMA = 2.211e5                                     # mu0 * gyromagnetic ratio of the electron (m/(A s))
Ms_f, t_f, alpha_f, Keff_f, P_f = 1.0e6, 1.3e-9, 0.01, 2.0e5, 0.6    # CoFeB-like perpendicular free layer
r_cell = 20e-9                                      # radius of the MTJ pillar (40 nm diameter)
HK_f = 2 * Keff_f / (MU0 * Ms_f)
eps_f = P_f / 2
Jc0 = alpha_f / eps_f * E * MU0 * Ms_f * t_f / HBAR * HK_f
V_f = np.pi * r_cell**2 * t_f
print(f"anisotropy field H_K = {HK_f / 1e3:.0f} kA/m (mu0 H_K = {MU0 * HK_f:.2f} T)")
print(f"critical current density J_c0 = {Jc0 / 1e10:.2f} MA/cm^2 -> current of a 40 nm cell {Jc0 * np.pi * r_cell**2 * 1e6:.0f} µA")
print(f"thermal stability of the bit K_eff V / kT = {Keff_f * V_f / (K_B * 300):.0f}  (> 60: data kept for > 10 years)")


def macrospin_stt(J_over_Jc, t_end=30e-9, dt=1e-12, theta0=0.05):
    """LLG + Slonczewski torque for a single moment (macrospin), all currents at once (RK4).
    Reference layer p = +z; the free layer starts antiparallel (near -z), tilted by theta0."""
    a_J = HBAR * eps_f * np.asarray(J_over_Jc, float)[:, None] * Jc0 / (E * MU0 * Ms_f * t_f)
    p = np.array([0.0, 0.0, 1.0])
    m = np.tile([np.sin(theta0), 0.0, -np.cos(theta0)], (len(a_J), 1))

    def rhs(m):
        h = HK_f * m[:, 2:3] * p + a_J * np.cross(m, p)          # anisotropy field + spin-torque field
        mxh = np.cross(m, h)
        return -GAMMA / (1 + alpha_f**2) * (mxh + alpha_f * np.cross(m, mxh))

    ts, mz = [0.0], [m[:, 2].copy()]
    for k in range(int(round(t_end / dt))):
        k1 = rhs(m); k2 = rhs(m + dt / 2 * k1); k3 = rhs(m + dt / 2 * k2); k4 = rhs(m + dt * k3)
        m = m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        if k % 10 == 9:
            ts.append((k + 1) * dt); mz.append(m[:, 2].copy())
    return np.array(ts), np.array(mz).T


t0 = time.time()
factors = np.array([0.9, 1.2, 1.5, 2.0, 3.0, 4.0])
ts, mz = macrospin_stt(factors)
print(f"computed in {time.time() - t0:.1f} s")
fig, ax = plt.subplots(figsize=(10, 3.8))
for f_, mzi, c in zip(factors, mz, plt.cm.viridis(np.linspace(0, 0.9, len(factors)))):
    ax.plot(ts * 1e9, mzi, color=c, lw=0.9, label=f"J = {f_} J$_{{c0}}$")
ax.set_xlabel("time (ns)"); ax.set_ylabel("m$_z$ of the free layer"); ax.set_ylim(-1.05, 1.05)
ax.set_title("macrospin: the current pumps up the precession until the free layer flips (AP → P); 0.9 J$_{c0}$ stays")
ax.legend(fontsize=8, ncol=3, loc="center right"); plt.tight_layout(); plt.show()

# %% [markdown]
# Below $J_{c0}$ the precession decays and the bit stays; above it the precession grows and the bit flips. The
# growth rate is proportional to the excess current, so the **switching time** is approximately
# $$t_{sw} \approx \frac{\tau_0}{J/J_{c0} - 1}\,\ln\frac{\pi}{2\theta_0},\qquad \tau_0 = \frac{1+\alpha^2}{\alpha\gamma H_K},$$
# where $\theta_0$ is the initial tilt of $\mathbf m$ from the easy axis. At $\theta_0 = 0$ the torque vanishes:
# in a real cell the initial tilt is provided by thermal fluctuations, which makes the switching time – and the
# **write error rate** – random.

# %%
fr = np.linspace(1.1, 5, 30)
ts2, mz2 = macrospin_stt(fr, t_end=40e-9)
t_sw = np.array([ts2[np.argmax(m_ > 0)] if (m_ > 0).any() else np.nan for m_ in mz2])
tau0 = (1 + alpha_f**2) / (alpha_f * GAMMA * HK_f)
plt.figure(figsize=(7, 3.6))
plt.plot(fr, t_sw * 1e9, "o", color=BLUE, label="macrospin simulation (first m$_z$ > 0)")
plt.plot(fr, tau0 * np.log(np.pi / (2 * 0.05)) / (fr - 1) * 1e9, color=ORANGE, label="formula")
plt.xlabel("J / J$_{c0}$"); plt.ylabel("switching time (ns)"); plt.ylim(0, 30); plt.legend()
plt.title(f"τ0 = {tau0 * 1e9:.2f} ns"); plt.tight_layout(); plt.show()

# %% [markdown]
# Writing faster costs current – a memory designer trades switching time against the write energy and against
# the breakdown voltage of the 1 nm MgO barrier.
#
# ### 22.4 The real cell is not a macrospin
#
# A 40 nm disc is several exchange lengths wide (section 7), so the magnetization need not stay uniform during the
# reversal. We simulated the same free layer with magnum.np – exchange, anisotropy, the full stray field and the
# Slonczewski torque on a grid of 2.5 nm cells – for four currents. Each run takes 5–10 minutes on a CPU, so the
# results were computed in advance (script `comsol_datasets/scripts/magnetism/stt_disc_magnumnp.py`); set
# `RUN_LIVE = True` to repeat one run here.

# %%
RUN_LIVE = False
disc_runs = {}
for f_ in (1.5, 2.0, 3.0, 4.0):
    disc_runs[f_] = dict(np.load(fetch(f"stt_disc_J{f_:.1f}.npz")))

if RUN_LIVE:                                         # ~5-10 min: repeat the run at 2 Jc0
    from magnumnp import SpinTorqueSlonczewski, LLGSolver, normalize
    torch.set_default_dtype(torch.float64)
    state = State(Mesh((16, 16, 1), (2.5e-9, 2.5e-9, t_f)))
    xx, yy, _ = state.SpatialCoordinate()
    disc = (xx - r_cell)**2 + (yy - r_cell)**2 < r_cell**2
    Msd, Ad, Kud = state.Constant(0.0), state.Constant(0.0), state.Constant(0.0)   # material only in the disc
    Msd[disc], Ad[disc], Kud[disc] = Ms_f, 1.5e-11, Keff_f + 0.5 * MU0 * Ms_f**2
    state.material = {"Ms": Msd, "A": Ad, "Ku": Kud, "Ku_axis": [0, 0, 1],
                      "alpha": alpha_f, "P": P_f, "Lambda": 1.0, "epsilon_prime": 0.0,
                      "mp": state.Constant([0, 0, 1.0]), "d": t_f, "J": 2.0 * Jc0}
    state.m = state.Constant([0.05, 0, -1.0]); normalize(state.m)
    llg = LLGSolver([DemagField(), ExchangeField(), UniaxialAnisotropyField(), SpinTorqueSlonczewski()])
    tl, ml = [0.0], [-1.0]
    while state.t < 15e-9:
        llg.step(state, 1e-11)
        tl.append(float(state.t)); ml.append(((state.m[..., 2] * disc).sum() / disc.sum()).item())
    disc_runs["live, 2.0"] = {"t": np.array(tl), "m": np.c_[0 * np.array(ml), 0 * np.array(ml), ml]}

fig, ax = plt.subplots(figsize=(10, 3.8))
tm, mzm = macrospin_stt(np.array([1.5, 2.0, 3.0, 4.0]), t_end=15e-9)
for (f_, run), mzi, c in zip(disc_runs.items(), mzm, (BLUE, ORANGE, "#2a9d4b", "#b03a2e")):
    ax.plot(run["t"] * 1e9, run["m"][:, 2], color=c, label=f"disc, J = {f_} J$_{{c0}}$")
    ax.plot(tm * 1e9, mzi, color=c, lw=0.6, ls=":")
for k in disc_runs:
    if isinstance(k, str):
        ax.plot(disc_runs[k]["t"] * 1e9, disc_runs[k]["m"][:, 2], "k--", lw=1, label="live run, 2 J$_{c0}$")
ax.set_xlabel("time (ns)"); ax.set_ylabel("average m$_z$"); ax.set_ylim(-1.05, 1.05)
ax.set_title("micromagnetic disc (solid) versus macrospin (dotted)")
ax.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()

# %%
run = disc_runs[2.0]
snaps, mask = run["mz_snap"].astype(float), run["mask"].astype(bool)
avg = run["m"][:, 2]
t_cross = run["t"][np.argmax(avg > 0)] if (avg > 0).any() else run["t"][-1] / 2
t_show = t_cross + np.array([-3.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]) * 1e-9   # around the zero crossing
idx = np.clip(np.round(t_show / float(run["dt_snap"])).astype(int), 0, len(snaps) - 1)
fig, ax = plt.subplots(1, len(idx), figsize=(14, 2.6))
for a, i in zip(ax, idx):
    a.imshow(np.where(mask, snaps[i], np.nan).T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
             extent=[0, 40, 0, 40])
    a.set_title(f"t = {i * float(run['dt_snap']) * 1e9:.2f} ns", fontsize=9); a.set_xticks([]); a.set_yticks([])
    a.grid(False)
fig.suptitle("m$_z$ of the free layer at 2 J$_{c0}$ (blue: −z, red: +z)", y=1.02); plt.show()

# %% [markdown]
# **What the micromagnetic model adds.**
# * **The real cell is stiffer.** The macrospin used the thin-film value $K_{eff} = K_u - \tfrac12\mu_0M_s^2$, valid
#   for an infinite film. A 40 nm disc has a demagnetizing factor $N_z < 1$ (section 2), so its stray-field energy
#   is smaller, its effective anisotropy larger, and so is its critical current. At 2 $J_{c0}$ the disc needs
#   7 ns instead of 4.6 ns; at 1.5 $J_{c0}$ (macrospin: 8 ns) it has not switched after 15 ns. At 4 $J_{c0}$ the
#   torque dominates and the two models agree (1.9 vs. 1.7 ns).
# * **The end of the reversal is not uniform.** The precession grows almost uniformly, but once it is large the
#   disc loses its coherence: a reversed domain **nucleates at the edge** and a **domain wall sweeps across** the
#   disc in less than a nanosecond (snapshots above) – the mechanisms of sections 11–15 again, now in a 40 nm memory
#   cell. Real cells add thermal fluctuations (random initial tilt), the stray field of the reference layer and the
#   heating by the current.
# * **Where the technology goes:** spin–orbit-torque MRAM (SOT, current in a heavy-metal line under the free layer –
#   separate read and write paths), voltage-controlled anisotropy, and the use of the same MTJs as nanoscale
#   oscillators, sensors and artificial neurons.

# %% [markdown]
# ## Summary
#
# | Section | What we learned |
# |---|---|
# | 1–3 | $\mathbf B = \mu_0(\mathbf H + \mathbf M)$; a uniform magnet has surface charges, inside it $H_d = -NM$, $\sum N = 1$ |
# | 4–5 | dia-, para-, ferro-, antiferro- and ferrimagnets differ in χ(T); the moments are electron spins, ~2.2 µB per Fe atom |
# | 6 | the exchange aligns the spins up to the Curie temperature |
# | 7 | four energy terms; micromagnetics resolves nanometres: $l_{ex}\approx 3.4$ nm, $\sqrt{A/K}\approx 21$ nm |
# | 8–10 | the stray field makes the long axis easy (shape anisotropy); uniform rotation gives a square loop; crystal anisotropy competes with the shape |
# | 11–13 | small particles are uniform; larger ones break into domains; the wall width is $\pi\sqrt{A/K}$ |
# | 14–15 | a field moves the wall, $v = \alpha\gamma\mu_0H\Delta/(1+\alpha^2)$; wall propagation needs a much smaller field than rotation – the depinning field sets $H_c$ |
# | 16–18 | the loop parameters separate soft and hard materials; the loop area gives the hysteresis loss; eddy currents add the classical and excess loss |
# | 19 | devices are simulated with measured B(H) and loss curves; two methods cross-check each other; DNN surrogates interpolate the simulations (and only interpolate) |
# | 20–21 | a permanent magnet works in the second quadrant, sized by $(BH)_{max}$ and its load line |
# | 22 | thermal stability ($K_uV/k_BT \gtrsim 60$) limits the grain size of recording media; GMR/TMR read the bits; a spin-polarized current above $J_{c0}$ writes an MRAM cell in nanoseconds |
#
# ## Sources and further reading
#
# * J. M. D. Coey, *Magnetism and Magnetic Materials*, Cambridge University Press 2010 – the main reference for
#   Part I, sections 9–12, 16, 20–22 and the material values.
# * S. Tumański, *Handbook of Magnetic Measurements*, CRC Press 2011 – soft magnetic materials and losses.
# * G. Bertotti, *Hysteresis in Magnetism*, Academic Press 1998 – loss separation.
# * A. Aharoni, *Introduction to the Theory of Ferromagnetism*, Oxford 2000 – demagnetizing factors, domains.
# * J. C. Slonczewski, *Current-driven excitation of magnetic multilayers*, J. Magn. Magn. Mater. 159, L1 (1996);
#   D. Apalkov et al., *Magnetoresistive random access memory*, Proc. IEEE 104, 1796 (2016) – spin-transfer torque
#   and MRAM.
# * F. Bruckner et al., *magnum.np – a PyTorch based GPU enhanced finite difference micromagnetic simulation
#   framework*, Sci. Rep. 13, 12054 (2023).
#
# All figures in this notebook are computed by the code above; numerical values are typical values from these
# books and producers' data sheets and should be checked against a data sheet before use in a design.
