"""2D DEVSIM model of the COMSOL library model bipolar_transistor_thermal (isothermal part).

Half device (symmetry at x = 0), 1.25 µm x 1 µm, out-of-plane length 3 µm. Physics as in COMSOL:
Boltzmann statistics, constant mobility 1450 / 500 cm²/Vs, SRH (tau = 10 µs, midgap), incomplete
ionization (ΔE = 50 meV, g_D = 2, g_A = 4), Nc, Nv ∝ T^1.5, Eg = 1.12 eV, eps_r = 11.7, ideal ohmic contacts.
"""
import contextlib, os, sys
import numpy as np
from scipy.optimize import brentq
import devsim
from devsim.python_packages.model_create import (
    CreateSolution, CreateNodeModel, CreateNodeModelDerivative, CreateEdgeModel, CreateEdgeModelDerivatives,
    CreateContactNodeModel)
from devsim.python_packages.simple_dd import CreateBernoulli, CreateElectronCurrent, CreateHoleCurrent

Q, KB, EPS0 = 1.602176634e-19, 1.380649e-23, 8.8541878128e-14
UM = 1e-4                                  # µm in cm
W, D, L_OUT = 1.25 * UM, 1.0 * UM, 3 * UM  # half width, thickness, out-of-plane length
W_CE, X_CB = 0.45 * UM, 0.95 * UM          # emitter contact 0..W_CE, base contact X_CB..W (top)
N_E, N_C = 2e20, 8e19
D_E, D_C = 0.15 * UM, 0.3 * UM


@contextlib.contextmanager
def quiet():
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


def graded(lines, f=1.0, ratio=1.15):
    """1D node positions through the given (position, spacing) points; spacing grows by at most `ratio`
    per step away from each point (like DEVSIM's add_*_mesh_line)."""
    pos = np.array([p for p, _ in lines]); ps = np.array([s for _, s in lines]) * f
    # target spacing at x: the smallest of ps_k * ratio^(distance/ps_k) over all points (approximately)
    def h(x):
        return np.min(ps + (ratio - 1) * np.abs(x - pos))
    out = []
    for a, b in zip(pos[:-1], pos[1:]):
        seg = [a]
        while seg[-1] < b:
            seg.append(seg[-1] + h(seg[-1]))
        seg = np.array(seg)
        seg = a + (seg - a) * (b - a) / (seg[-1] - a)  # stretch so the segment ends exactly at b
        out.extend(seg[:-1])
    out.append(pos[-1])
    return np.array(out)


def gauss_len(n0, nb, depth):
    """Gaussian decay length so that n0*exp(-(depth/l)^2) = nb (COMSOL 'junction depth' option)."""
    return depth / np.sqrt(np.log(n0 / nb))


def outside(coord, lo, hi):
    """Distance outside the interval [lo, hi] (0 inside), as a DEVSIM expression."""
    return f"(ifelse({coord}<{lo},{lo}-{coord},0)+ifelse({coord}>{hi},{coord}-{hi},0))"


def doping(NB, Nepi):
    """Donor and acceptor expressions = COMSOL analytic doping features adm1..adm4."""
    nb_box = NB + Nepi
    l_b = gauss_len(nb_box, Nepi, D_E)             # base: box y in [-d_E, 0], to the existing doping N_epi at d_E
    l_e = gauss_len(N_E + NB, NB, D_E)              # emitter: line box x in [0, w_cE], y = 0, to N_B at d_E
    l_c = gauss_len(N_C, Nepi, 1.3 * D_C)           # buried collector: box below the device, to N_epi
    # DEVSIM parses -a^2 as (-a)^2, so squares are written as products
    def g(d, l):
        return f"exp(-1*({d}/{l})*({d}/{l}))"
    acc = f"{nb_box}*{g(outside('y', -D_E, 0), l_b)}"
    don_e = f"{N_E + NB}*{g(outside('x', 0, W_CE), l_e)}*{g(outside('y', 0, 0), l_e)}"
    don_c = f"{N_C}*{g(outside('y', -D - D_C, -D), l_c)}"
    return f"{Nepi}+{don_e}+{don_c}", acc


def contact_equilibrium(nd, na, T):
    """n, p and intrinsic-referenced potential of a neutral region with incomplete ionization."""
    kT = KB * T / Q
    Nc, Nv = 2.8e19 * (T / 300) ** 1.5, 1.04e19 * (T / 300) ** 1.5
    ni = np.sqrt(Nc * Nv) * np.exp(-1.12 / (2 * kT))
    n1d, p1a = Nc * np.exp(-0.05 / kT), Nv * np.exp(-0.05 / kT)

    def charge(ln_n):
        n = np.exp(ln_n); p = ni ** 2 / n
        return p - n + nd / (1 + 2 * n / n1d) - na / (1 + 4 * p / p1a)
    n = np.exp(brentq(charge, np.log(1e-30), np.log(1e22), xtol=1e-14))
    return n, ni ** 2 / n, kT * np.log(n / ni)


_count = [0]


class BJT:
    def __init__(self, T, NB, Nepi, fine=1.0):
        _count[0] += 1
        self.dev, self.reg, mesh = f"bjt{_count[0]}", "si", f"m{_count[0]}"
        self.T = T
        with quiet():
            self._build(mesh, NB, Nepi, fine)

    def _build(self, mesh, NB, Nepi, f):
        dev, reg = self.dev, self.reg
        xs = graded([(0, 40e-7), (W_CE, 8e-7), (0.6 * UM, 4e-7), (X_CB, 10e-7), (W, 20e-7)], f)
        ys = graded([(-D, 6e-7), (-0.75 * UM, 15e-7), (-0.27 * UM, 2e-7), (-0.15 * UM, 1.5e-7), (0, 3e-7)], f)
        self.nx, self.ny = len(xs), len(ys)
        nx = len(xs)
        X, Y = np.meshgrid(xs, ys)  # node k = j*nx + i
        coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).ravel().tolist()
        names = [reg, "emitter", "base", "collector"]
        el = []
        for j in range(len(ys) - 1):
            for i in range(nx - 1):
                a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i
                el += [2, 0, a, b, c, 2, 0, a, c, d]
        top = (len(ys) - 1) * nx
        for i in range(nx - 1):
            if xs[i + 1] <= W_CE * (1 + 1e-9):
                el += [1, 1, top + i, top + i + 1]
            if xs[i] >= X_CB * (1 - 1e-9):
                el += [1, 2, top + i, top + i + 1]
            el += [1, 3, i, i + 1]
        devsim.create_gmsh_mesh(mesh=mesh, coordinates=coords, elements=el, physical_names=names)
        devsim.add_gmsh_region(mesh=mesh, gmsh_name=reg, region=reg, material="Si")
        for c in names[1:]:
            devsim.add_gmsh_contact(mesh=mesh, gmsh_name=c, name=c, region=reg, material="metal")
        devsim.finalize_mesh(mesh=mesh)
        devsim.create_device(mesh=mesh, device=dev)

        T = self.T
        kT = KB * T / Q
        Nc, Nv = 2.8e19 * (T / 300) ** 1.5, 1.04e19 * (T / 300) ** 1.5
        ni = np.sqrt(Nc * Nv) * np.exp(-1.12 / (2 * kT))
        for name, value in [("Permittivity", 11.7 * EPS0), ("ElectronCharge", Q), ("n_i", ni), ("V_t", kT),
                            ("n1", ni), ("p1", ni), ("taun", 1e-5), ("taup", 1e-5), ("mu_n", 1450), ("mu_p", 500),
                            ("n1d", Nc * np.exp(-0.05 / kT)), ("p1a", Nv * np.exp(-0.05 / kT))]:
            devsim.set_parameter(device=dev, region=reg, name=name, value=value)
        don, acc = doping(NB, Nepi)
        CreateNodeModel(dev, reg, "Donors", don)
        CreateNodeModel(dev, reg, "Acceptors", acc)
        # contacts: uniform doping along each contact
        self.eq = {}
        for c, x, y in [("emitter", 0.2 * UM, 0), ("base", 1.1 * UM, 0), ("collector", 0.6 * UM, -D)]:
            nd, na = self._eval_doping(don, acc, x, y)
            self.eq[c] = contact_equilibrium(nd, na, T)
            n0, p0, psi0 = self.eq[c]
            for k, v in (("n0", n0), ("p0", p0), ("psi0", psi0), ("bias", 0.0)):
                devsim.set_parameter(device=dev, name=f"{c}_{k}", value=v)

        # --- Poisson only (equilibrium carriers from the potential)
        CreateSolution(dev, reg, "Potential")
        ion = "Donors/(1+2*{n}/n1d)-Acceptors/(1+4*{p}/p1a)"
        for name, expr in [("IntrinsicElectrons", "n_i*exp(Potential/V_t)"), ("IntrinsicHoles", "n_i^2/IntrinsicElectrons"),
                           ("IntrinsicCharge", "IntrinsicHoles-IntrinsicElectrons+" + ion.format(n="IntrinsicElectrons", p="IntrinsicHoles")),
                           ("PotentialIntrinsicCharge", "-ElectronCharge*IntrinsicCharge")]:
            CreateNodeModel(dev, reg, name, expr)
            CreateNodeModelDerivative(dev, reg, name, expr, "Potential")
        for name, expr in [("ElectricField", "(Potential@n0-Potential@n1)*EdgeInverseLength"),
                           ("PotentialEdgeFlux", "Permittivity*ElectricField")]:
            CreateEdgeModel(dev, reg, name, expr)
            CreateEdgeModelDerivatives(dev, reg, name, expr, "Potential")
        devsim.equation(device=dev, region=reg, name="PotentialEquation", variable_name="Potential",
                        node_model="PotentialIntrinsicCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
        CreateEdgeModel(dev, reg, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(dev, reg, "contactcharge_edge", "Permittivity*ElectricField", "Potential")
        for c in self.eq:
            CreateContactNodeModel(dev, c, f"{c}_pot", f"Potential-{c}_bias-{c}_psi0")
            CreateContactNodeModel(dev, c, f"{c}_pot:Potential", "1")
            devsim.contact_equation(device=dev, contact=c, name="PotentialEquation", node_model=f"{c}_pot",
                                    edge_charge_model="contactcharge_edge")
        self.solve(abs_err=1.0)

        # --- drift-diffusion
        CreateSolution(dev, reg, "Electrons")
        CreateSolution(dev, reg, "Holes")
        devsim.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
        devsim.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
        pnc = "-ElectronCharge*(Holes-Electrons+" + ion.format(n="Electrons", p="Holes") + ")"
        CreateNodeModel(dev, reg, "PotentialNodeCharge", pnc)
        for v in ("Electrons", "Holes"):
            CreateNodeModelDerivative(dev, reg, "PotentialNodeCharge", pnc, v)
        devsim.equation(device=dev, region=reg, name="PotentialEquation", variable_name="Potential",
                        node_model="PotentialNodeCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
        CreateBernoulli(dev, reg)
        usrh = "(Electrons*Holes-n_i^2)/(taup*(Electrons+n1)+taun*(Holes+p1))"
        for name, expr in [("USRH", usrh), ("ElectronGeneration", "-ElectronCharge*USRH"), ("HoleGeneration", "ElectronCharge*USRH")]:
            CreateNodeModel(dev, reg, name, expr)
            for v in ("Electrons", "Holes"):
                CreateNodeModelDerivative(dev, reg, name, expr, v)
        CreateElectronCurrent(dev, reg, "mu_n")
        CreateHoleCurrent(dev, reg, "mu_p")
        devsim.equation(device=dev, region=reg, name="ElectronContinuityEquation", variable_name="Electrons",
                        edge_model="ElectronCurrent", node_model="ElectronGeneration", variable_update="positive")
        devsim.equation(device=dev, region=reg, name="HoleContinuityEquation", variable_name="Holes",
                        edge_model="HoleCurrent", node_model="HoleGeneration", variable_update="positive")
        for c in self.eq:
            for v, cur in (("Electrons", "ElectronCurrent"), ("Holes", "HoleCurrent")):
                ref = f"{c}_n0" if v == "Electrons" else f"{c}_p0"
                CreateContactNodeModel(dev, c, f"{c}_{v}", f"{v}-{ref}")
                CreateContactNodeModel(dev, c, f"{c}_{v}:{v}", "1")
                eqn = "ElectronContinuityEquation" if v == "Electrons" else "HoleContinuityEquation"
                devsim.contact_equation(device=dev, contact=c, name=eqn, node_model=f"{c}_{v}", edge_current_model=cur)
        self.solve()

    def _eval_doping(self, don, acc, x, y):
        ex = {"exp": np.exp, "ifelse": lambda c, a, b: a if c else b, "x": x, "y": y}
        return eval(don.replace("^", "**"), ex), eval(acc.replace("^", "**"), ex)

    def solve(self, abs_err=1e10):
        devsim.solve(type="dc", absolute_error=abs_err, relative_error=1e-10, maximum_iterations=50)

    def set_bias(self, c, v):
        devsim.set_parameter(device=self.dev, name=f"{c}_bias", value=v)

    def get_bias(self, c):
        return devsim.get_parameter(device=self.dev, name=f"{c}_bias")

    def ramp(self, c, v, step=0.1):
        """Moves contact c to v in steps, halving the step when Newton fails."""
        with quiet():
            now = self.get_bias(c)
            while abs(v - now) > 1e-12:
                s = np.sign(v - now) * min(step, abs(v - now))
                self.set_bias(c, now + s)
                try:
                    self.solve()
                    now += s
                except devsim.error:
                    step /= 2
                    if step < 1e-4:
                        self.set_bias(c, now)
                        raise
                    continue

    def current(self, c):
        """Current into the contact (A) for the 3 µm long device, COMSOL sign convention."""
        i = sum(devsim.get_contact_current(device=self.dev, contact=c, equation=e)
                for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))
        return i * L_OUT

    def currents(self):
        return {c: self.current(c) for c in ("emitter", "base", "collector")}

    def close(self):
        devsim.delete_device(device=self.dev)
