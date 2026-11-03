"""2D DEVSIM model of the COMSOL library model bipolar_transistor_thermal, with self-heating.

Same device and physics as devsim_bjt.py, but the temperature is a field T(x, y):
* DEVSIM solves the drift-diffusion equations for a given T(x, y) (Nc, Nv, n_i, ionization and V_t
  depend on the local temperature; the Scharfetter-Gummel currents include thermal diffusion,
  J_n = q mu_n (n grad psi + V_t grad n + n grad(V_t)/2), as in COMSOL (D_th = mu k T / 2q)).
* Our own finite-volume solver (scipy, on the same mesh) solves the steady heat equation
      div(k grad T) + H = 0,   -k dT/dn = h0 (T - T0) on the bottom, emitter, base and right side,
  with h0 = 1 / (R * (w_cE + w_cB + w_cC) * l_BJT) as in COMSOL, and the insulated left (symmetry) side
  and top between the contacts. H is the dissipated power density: Joule heat J.grad(quasi-Fermi
  potential) on the mesh edges plus recombination heat q*U*(phi_p - phi_n); its integral is exactly the
  electrical power I*V entering the contacts.
* The two are coupled by fixed-point iteration (solve DD -> heat -> new T -> ...).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import devsim
from devsim.python_packages.model_create import (
    CreateSolution, CreateNodeModel, CreateNodeModelDerivative, CreateEdgeModel, CreateEdgeModelDerivatives,
    CreateContactNodeModel)
from devsim_bjt import Q, KB, EPS0, UM, W, D, L_OUT, W_CE, X_CB, graded, doping, quiet

EG, DE = 1.12, 0.05             # band gap and dopant ionization energy (eV)
K_SI = 1.31                     # thermal conductivity (W/(cm K))
A_CONTACTS = (0.45 + 0.30 + 1.25) * UM * L_OUT   # (w_cE + w_cB + w_cC) * l_BJT in cm^2

# temperature-dependent quantities as DEVSIM expressions of the node solution Temperature
VT = "(kq*Temperature)"
NC = "(2.8e19*pow(Temperature/300,1.5))"
NV = "(1.04e19*pow(Temperature/300,1.5))"
NI2 = f"({NC}*{NV}*exp(-Eg/{VT}))"
NI = f"pow({NI2},0.5)"
N1D = f"({NC}*exp(-dE/{VT}))"
P1A = f"({NV}*exp(-dE/{VT}))"
VTE = "(kq*0.5*(Temperature@n0+Temperature@n1))"


def neutral(nd, na, T):
    """Equilibrium n, p of a neutral region with incomplete ionization at temperature T."""
    kT = KB * T / Q
    Nc, Nv = 2.8e19 * (T / 300) ** 1.5, 1.04e19 * (T / 300) ** 1.5
    ni2 = Nc * Nv * np.exp(-EG / kT)
    n1d, p1a = Nc * np.exp(-DE / kT), Nv * np.exp(-DE / kT)

    def charge(ln_n):
        n = np.exp(ln_n); p = ni2 / n
        return p - n + nd / (1 + 2 * n / n1d) - na / (1 + 4 * p / p1a)
    n = np.exp(brentq(charge, np.log(1e-30), np.log(1e22), xtol=1e-14))
    return n, ni2 / n


_count = [0]


class BJTThermal:
    def __init__(self, T0, NB, Nepi, R=100.0, fine=1.0, thermodiff=0.5):
        _count[0] += 1
        self.dev, self.reg, mesh = f"bjtt{_count[0]}", "si", f"mt{_count[0]}"
        self.T0, self.R, self.thermodiff = T0, R, thermodiff
        with quiet():
            self._build(mesh, NB, Nepi, fine)
        self._heat_setup()

    # ------------------------------------------------------------------ semiconductor
    def _build(self, mesh, NB, Nepi, f):
        dev, reg = self.dev, self.reg
        xs = graded([(0, 40e-7), (W_CE, 8e-7), (0.6 * UM, 4e-7), (X_CB, 10e-7), (W, 20e-7)], f)
        ys = graded([(-D, 6e-7), (-0.75 * UM, 15e-7), (-0.27 * UM, 2e-7), (-0.15 * UM, 1.5e-7), (0, 3e-7)], f)
        nx = len(xs)
        X, Y = np.meshgrid(xs, ys)
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

        for name, value in [("Permittivity", 11.7 * EPS0), ("ElectronCharge", Q), ("kq", KB / Q), ("Eg", EG),
                            ("dE", DE), ("taun", 1e-5), ("taup", 1e-5), ("mu_n", 1450), ("mu_p", 500)]:
            devsim.set_parameter(device=dev, region=reg, name=name, value=value)
        don, acc = doping(NB, Nepi)
        CreateNodeModel(dev, reg, "Donors", don)
        CreateNodeModel(dev, reg, "Acceptors", acc)
        self.Nd = np.array(devsim.get_node_model_values(device=dev, region=reg, name="Donors"))
        self.Na = np.array(devsim.get_node_model_values(device=dev, region=reg, name="Acceptors"))
        self.x = np.array(devsim.get_node_model_values(device=dev, region=reg, name="x"))
        self.y = np.array(devsim.get_node_model_values(device=dev, region=reg, name="y"))
        self.nn = len(self.x)

        CreateSolution(dev, reg, "Temperature")   # a field we set ourselves (no equation in DEVSIM)
        devsim.set_node_values(device=dev, region=reg, name="Temperature", values=[self.T0] * self.nn)
        # contact boundary values per node (they depend on the local contact temperature)
        self.cnodes = {c: np.array(devsim.get_node_model_values(device=dev, region=reg, name="node_index"))[0:0]
                       for c in ("emitter", "base", "collector")}
        for c in self.cnodes:
            idx = devsim.get_element_node_list(device=dev, region=reg, contact=c)
            self.cnodes[c] = np.unique(np.array(idx).ravel())
            devsim.set_parameter(device=dev, name=f"{c}_bias", value=0.0)
        for v in ("BC_n", "BC_p", "BC_psi"):
            CreateSolution(dev, reg, v)
        self._set_contact_values()

        # --- Poisson only: equilibrium carriers from the potential (Fermi level 0)
        # band edges: Ec = -psi + Eg/2, Ev = -psi - Eg/2 (eV), so n = Nc exp((psi - Eg/2 - V)/Vt)
        CreateSolution(dev, reg, "Potential")
        ion = f"Donors/(1+2*{{n}}/{N1D})-Acceptors/(1+4*{{p}}/{P1A})"
        for name, expr in [("IntrinsicElectrons", f"{NC}*exp((Potential-0.5*Eg)/{VT})"),
                           ("IntrinsicHoles", f"{NV}*exp((-Potential-0.5*Eg)/{VT})"),
                           ("IntrinsicCharge", "IntrinsicHoles-IntrinsicElectrons+" + ion.format(n="IntrinsicElectrons", p="IntrinsicHoles")),
                           ("PotentialIntrinsicCharge", "-ElectronCharge*IntrinsicCharge")]:
            CreateNodeModel(dev, reg, name, expr)
            CreateNodeModelDerivative(dev, reg, name, expr, "Potential")
        devsim.set_node_values(device=dev, region=reg, name="Potential", init_from="BC_psi")
        for name, expr in [("ElectricField", "(Potential@n0-Potential@n1)*EdgeInverseLength"),
                           ("PotentialEdgeFlux", "Permittivity*ElectricField")]:
            CreateEdgeModel(dev, reg, name, expr)
            CreateEdgeModelDerivatives(dev, reg, name, expr, "Potential")
        devsim.equation(device=dev, region=reg, name="PotentialEquation", variable_name="Potential",
                        node_model="PotentialIntrinsicCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
        CreateEdgeModel(dev, reg, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(dev, reg, "contactcharge_edge", "Permittivity*ElectricField", "Potential")
        for c in self.cnodes:
            CreateContactNodeModel(dev, c, f"{c}_pot", f"Potential-{c}_bias-BC_psi")
            CreateContactNodeModel(dev, c, f"{c}_pot:Potential", "1")
            devsim.contact_equation(device=dev, contact=c, name="PotentialEquation", node_model=f"{c}_pot",
                                    edge_charge_model="contactcharge_edge")
        self.solve(abs_err=1.0)

        # --- drift-diffusion with local temperature
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
        usrh = f"(Electrons*Holes-{NI2})/(taup*(Electrons+{NI})+taun*(Holes+{NI}))"
        for name, expr in [("USRH", usrh), ("ElectronGeneration", "-ElectronCharge*USRH"), ("HoleGeneration", "ElectronCharge*USRH")]:
            CreateNodeModel(dev, reg, name, expr)
            for v in ("Electrons", "Holes"):
                CreateNodeModelDerivative(dev, reg, name, expr, v)
        # Scharfetter-Gummel with edge-averaged V_t for J = q mu (n grad(psi) + grad(n V_t))
        xv = f"((Potential@n0-Potential@n1)/{VTE})"
        # J = q mu (n grad psi + V_t grad n + c n grad V_t) = q mu V_t^(1-c) [grad(n V_t^c) + n V_t^c grad(psi)/V_t];
        # c = thermodiff is the thermal diffusion coefficient in units of mu*k/q (COMSOL: D_th = mu*k*T/(2q), c = 1/2)
        c = self.thermodiff
        pre = f"pow({VTE},{1 - c})"
        t0, t1 = f"pow(kq*Temperature@n0,{c})", f"pow(kq*Temperature@n1,{c})"
        jn = (f"ElectronCharge*mu_n*EdgeInverseLength*{pre}*(Electrons@n1*{t1}*B(-1*{xv})"
              f"-Electrons@n0*{t0}*B({xv}))")
        jp = (f"-ElectronCharge*mu_p*EdgeInverseLength*{pre}*(Holes@n1*{t1}*B({xv})"
              f"-Holes@n0*{t0}*B(-1*{xv}))")
        for name, expr in (("ElectronCurrent", jn), ("HoleCurrent", jp)):
            CreateEdgeModel(dev, reg, name, expr)
            for v in ("Potential", "Electrons", "Holes"):
                CreateEdgeModelDerivatives(dev, reg, name, expr, v)
        devsim.equation(device=dev, region=reg, name="ElectronContinuityEquation", variable_name="Electrons",
                        edge_model="ElectronCurrent", node_model="ElectronGeneration", variable_update="positive")
        devsim.equation(device=dev, region=reg, name="HoleContinuityEquation", variable_name="Holes",
                        edge_model="HoleCurrent", node_model="HoleGeneration", variable_update="positive")
        for c in self.cnodes:
            for v, cur, ref in (("Electrons", "ElectronCurrent", "BC_n"), ("Holes", "HoleCurrent", "BC_p")):
                CreateContactNodeModel(dev, c, f"{c}_{v}", f"{v}-{ref}")
                CreateContactNodeModel(dev, c, f"{c}_{v}:{v}", "1")
                eqn = "ElectronContinuityEquation" if v == "Electrons" else "HoleContinuityEquation"
                devsim.contact_equation(device=dev, contact=c, name=eqn, node_model=f"{c}_{v}", edge_current_model=cur)
        # quasi-Fermi potentials (V) for the heat source
        CreateNodeModel(dev, reg, "PhiN", f"Potential-0.5*Eg-{VT}*log(Electrons/{NC})")
        CreateNodeModel(dev, reg, "PhiP", f"Potential+0.5*Eg+{VT}*log(Holes/{NV})")
        devsim.edge_from_node_model(device=dev, region=reg, node_model="node_index")
        self.solve()

    def _set_contact_values(self):
        """Equilibrium n, p and potential at each contact node for its current temperature."""
        T = self.temperature()
        bn, bp, bpsi = (np.zeros(self.nn) for _ in range(3))
        for c, nodes in self.cnodes.items():
            for k in nodes:
                n, p = neutral(self.Nd[k], self.Na[k], T[k])
                Nc = 2.8e19 * (T[k] / 300) ** 1.5
                bn[k], bp[k], bpsi[k] = n, p, 0.5 * EG + KB * T[k] / Q * np.log(n / Nc)
        for name, v in (("BC_n", bn), ("BC_p", bp), ("BC_psi", bpsi)):
            devsim.set_node_values(device=self.dev, region=self.reg, name=name, values=v.tolist())

    # ------------------------------------------------------------------ heat
    def _heat_setup(self):
        """Finite-volume conduction matrix on the DEVSIM mesh, with the convective boundary terms."""
        g = lambda m: np.array(devsim.get_edge_model_values(device=self.dev, region=self.reg, name=m))
        self.e0, self.e1 = g("node_index@n0").astype(int), g("node_index@n1").astype(int)
        self.ecouple = g("EdgeCouple")
        cond = K_SI * self.ecouple * g("EdgeInverseLength")      # W/(cm K) per unit depth
        n, e0, e1 = self.nn, self.e0, self.e1
        A = sp.coo_matrix((np.r_[cond, cond, -cond, -cond], (np.r_[e0, e1, e0, e1], np.r_[e0, e1, e1, e0])), shape=(n, n))
        # boundary length of each node on the cooled boundaries (half of each adjacent boundary segment)
        x, y, eps = self.x, self.y, 1e-9
        blen = np.zeros(n)
        def add_segments(mask, coord, keep):
            idx = np.where(mask)[0]
            idx = idx[np.argsort(coord[idx])]
            for a, b in zip(idx[:-1], idx[1:]):
                if keep(coord[a], coord[b]):
                    h = abs(coord[b] - coord[a]) / 2
                    blen[a] += h; blen[b] += h
        add_segments(np.abs(y + D) < eps, x, lambda a, b: True)                                  # collector (bottom)
        add_segments(np.abs(y) < eps, x, lambda a, b: b <= W_CE * (1 + eps) or a >= X_CB * (1 - eps))  # emitter, base
        add_segments(np.abs(x - W) < eps, y, lambda a, b: True)                                   # right side
        self.h0 = 1.0 / (self.R * A_CONTACTS)                    # W/(cm^2 K)
        self.hb = self.h0 * blen
        self.Kmat = (A + sp.diags(self.hb)).tocsc()
        self.Klu = spla.splu(self.Kmat)
        self.volume = np.array(devsim.get_node_model_values(device=self.dev, region=self.reg, name="NodeVolume"))

    def heat_power(self):
        """Dissipated power per node (W per cm of depth): Joule heat on edges + recombination heat."""
        g = lambda m: np.array(devsim.get_edge_model_values(device=self.dev, region=self.reg, name=m))
        nv = lambda m: np.array(devsim.get_node_model_values(device=self.dev, region=self.reg, name=m))
        phin, phip = nv("PhiN"), nv("PhiP")
        e0, e1 = self.e0, self.e1
        pe = self.ecouple * (g("ElectronCurrent") * (phin[e0] - phin[e1]) + g("HoleCurrent") * (phip[e0] - phip[e1]))
        P = np.zeros(self.nn)
        np.add.at(P, e0, pe / 2)
        np.add.at(P, e1, pe / 2)
        P += Q * nv("USRH") * (phip - phin) * self.volume
        return P

    def temperature(self):
        return np.array(devsim.get_node_model_values(device=self.dev, region=self.reg, name="Temperature"))

    def set_temperature(self, T):
        devsim.set_node_values(device=self.dev, region=self.reg, name="Temperature", values=list(T))
        self._set_contact_values()

    def solve_heat(self):
        return self.Klu.solve(self.heat_power() + self.hb * self.T0)

    def self_consistent(self, tol=1e-3, maxit=60, relax=1.0):
        """Fixed-point iteration DD <-> heat at the present bias; returns the number of iterations."""
        T = self.temperature()
        for it in range(maxit):
            Tn = self.solve_heat()
            dT = np.max(np.abs(Tn - T))
            T = T + relax * (Tn - T)
            with quiet():
                self.set_temperature(T)
                self.solve()
            if dT < tol:
                return it + 1
        raise RuntimeError(f"thermal iteration did not converge (last change {dT:.3g} K)")

    # ------------------------------------------------------------------ bias
    def solve(self, abs_err=1e10):
        devsim.solve(type="dc", absolute_error=abs_err, relative_error=1e-10, maximum_iterations=50)

    def set_bias(self, c, v):
        devsim.set_parameter(device=self.dev, name=f"{c}_bias", value=v)

    def get_bias(self, c):
        return devsim.get_parameter(device=self.dev, name=f"{c}_bias")

    def ramp(self, c, v, step=0.1, thermal=False):
        """Moves contact c to v in steps (halving the step when Newton fails); with thermal=True the
        temperature is made self-consistent after every step."""
        now = self.get_bias(c)
        while abs(v - now) > 1e-12:
            s = np.sign(v - now) * min(step, abs(v - now))
            T_saved = self.temperature()
            self.set_bias(c, now + s)
            try:
                with quiet():
                    self.solve()
                if thermal:
                    self.self_consistent()
                now += s
            except (devsim.error, RuntimeError):
                step /= 2
                with quiet():
                    self.set_temperature(T_saved)
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

    def power(self):
        """Electrical power entering the contacts (W)."""
        return sum(self.get_bias(c) * self.current(c) for c in ("emitter", "base", "collector"))

    def close(self):
        devsim.delete_device(device=self.dev)
