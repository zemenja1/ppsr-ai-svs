# COMSOL semiconductor datasets for DNN training

Training data for the teaching project *AI ve výuce – Elektrotechnologie* (CTU FEL).
All data were generated with **COMSOL Multiphysics 6.3 + Semiconductor Module** in batch mode,
starting from models of the COMSOL Application Library. The CSV files are meant to be put on
GitHub and loaded directly into Google Colab.

| Case study | Model | Inputs | Output |
|---|---|---|---|
| 1. pn diode (1D) | `pn_junction_1d` | Na, Nd, carrier lifetime, temperature, bias V | current I, current density J |
| 2. Silicon solar cell (1D) | `si_solar_cell_1d` | base/emitter doping, emitter depth, thickness, lifetime, irradiance, temperature, voltage V | current density J (+ Jsc, Voc, FF, η per cell) |
| 3. MOSFET (2D) | `mosfet` | Vg, Vd (tier 1); + NA, tox, Lg, junction depth, temperature (tier 2) | drain current Id |
| 4. Bipolar transistor with self-heating (2D) | `bipolar_transistor_thermal` | ambient temperature, thermal resistance (cooling), base and epi doping; base voltage and collector voltage (Gummel) or base current and collector voltage (output characteristics) | emitter/base/collector currents, V_BE, maximum temperature |

The models are from the COMSOL Application Library (Semiconductor Module). The `.mph` files are
licensed under the COMSOL Software License Agreement and are **not** included in this repository;
`scripts/setup_models.sh` copies them from your own COMSOL installation.

## Contents of the current version (v7, COMSOL 6.3, ~55 h in total on 12 CPU cores)

| File | Rows | Contents |
|---|---|---|
| `mosfet/mosfet_tier1_grid.csv` | 336 | 21 Vg × 16 Vd |
| `mosfet/mosfet_tier1_idvg.csv` | 51 | Vg = 0…5 V, step 0.1 V |
| `mosfet/mosfet_tier2.csv` | 7 854 | 154 devices × (11 Id–Vg + 40 grid points) |
| `mosfet/mosfet_tier2_devices.csv` | 156 | 154 `ok`, 2 `failed` (see below) |
| `mosfet/mosfet_tier2_idvg_fine.csv` | 4 580 | 154 devices × 25–32 Vg points (dense near the threshold) |
| `pn_diode/pn_diode_iv.csv` | 11 500 | 500 diodes × 23 voltages |
| `solar_cell/solar_cell_jv.csv` | 11 700 | 450 cells × 26 voltages |
| `solar_cell/solar_cell_summary.csv` | 450 | one row per cell |
| `bjt_thermal/bjt_thermal_gummel.csv` | 8 773 | 400 samples × (11 isothermal + 11 self-heating Gummel points), 27 missing |
| `bjt_thermal/bjt_thermal_samples.csv` | 400 | 395 `ok`, 5 `partial` (see below) |
| `bjt_thermal/bjt_thermal_output.csv` | 22 643 | 397 samples × 3 base currents × 10 collector voltages × 2 (isothermal, self-heating), 1 177 missing |
| `bjt_thermal/bjt_thermal_output_samples.csv` | 400 | 337 `ok`, 60 `partial`, 3 `failed` (see below) |

Known gaps:
* MOSFET devices `m0055` and `m0067` (NA ≈ 2.6e17 cm⁻³ at T ≈ 282–286 K) did not converge at the first
  bias point and have no data, so the corner *high doping + low temperature* is slightly under-sampled.
* Tier 2 consists of the 150-device Latin hypercube (`m0001`–`m0150`) plus the first 6 devices of a
  second hypercube (`m0151`–`m0156`); the second batch was cut by the time budget.
* The solar cells are the first 450 of a 600-cell Latin hypercube (cut by the time budget; because
  the sample order is random, the 450 cells still cover the whole parameter space).
* Bipolar transistor, Gummel plots: all 400 samples were computed. 5 samples are incomplete: `b0013`
  and `b0019` have no self-heating points (the coupled study did not converge), `b0022`, `b0148` and
  `b0308` miss the top 1–2 isothermal points.
* Bipolar transistor, output characteristics: `b0022`, `b0148` and `b0308` failed in the initialization
  (the same samples are incomplete in the Gummel data). The job of `b0312` hit the time limit during
  its last curve (self-heating, 1 mA), 3 points are missing. In 59 other samples one or two curves are missing – almost always the whole curve of one base
  current, 100 µA or 1 mA, isothermal and therefore also self-heating (the self-heating sweep starts
  from the isothermal one). These failures are more frequent at low T0 and high collector voltage of
  the starting point, so **high base current at low ambient temperature is under-sampled**.

### Version history
* **v1:** in the pn diode and solar cell data the `tau_s` column had **no
  effect** – the COMSOL recombination features took the lifetime from the material, so the value set by
  the generator was ignored (diode: τ = 1 µs, solar cell: τ = 10 µs for all devices). In the diode data
  nᵢ was also independent of temperature, so currents *decreased* with T. Found when comparing COMSOL
  with DEVSIM; both datasets were regenerated.
* **v2:** lifetime and nᵢ(T) corrected. But in ~60 % of the diode curves one
  reverse-bias point (and in ~1 % of the solar J–V curves one low-voltage point) was wrong: COMSOL's
  default convergence test checks only the size of the Newton update, and accepted iterations that had
  stalled with a large residual – without any warning. Found by checking the curves for smoothness.
* **v3:** pn diode and solar cell data regenerated with `termonres = on` (the residual must
  be small, too) for all nonlinear solvers. Reverse currents now agree with DEVSIM to ~4 digits above
  ~1e-7 A/cm². All 500 diode I–V curves are monotonic. In 13 solar J–V curves J rises by < 3e-5
  (relative) between neighbouring points on the flat plateau below the maximum power point – solver
  tolerance, not an error. The MOSFET data were checked (all Id(Vd) and Id(Vg) curves monotonic) and
  not regenerated.
* **v4:** added `mosfet_tier2_idvg_fine.csv` (dense Id–Vg sweeps of all tier 2
  devices, agree with the 0.5 V points of `mosfet_tier2.csv` to 2e-5 relative) and a 2D diode with self-heating
  (removed in v7). Existing files unchanged.
* **v5:** added the bipolar transistor with self-heating (`bjt_thermal/`), computed
  with the studies of the Application Library model *bipolar_transistor_thermal*. Existing files unchanged.
* **v6:** bipolar transistor: the remaining 124 Gummel samples (`b0277`–`b0400`), and the
  output characteristics (`bjt_thermal_output.csv`) of the same 400 samples. The Gummel parser now keeps
  only the 11 grid points of V_B (two partial curves had contained points of a failed step-size
  reduction).
* **v7:** removed the diode with self-heating (`thermal_diode/`, lumped thermal model); the bipolar
  transistor is the self-heating case study.

## Files

```
data/
  mosfet/
    mosfet_tier1_grid.csv       reference device, dense Vg x Vd grid
    mosfet_tier1_idvg.csv       reference device, Id-Vg at Vd = 10 mV (threshold region, fine Vg steps)
    mosfet_tier2.csv            many devices (Latin hypercube), Id-Vg at Vd = 10 mV + Vg x Vd grid
    mosfet_tier2_devices.csv    one row per device: parameters, status, number of points, runtime
    mosfet_tier2_idvg_fine.csv  tier 2 devices, Id-Vg at Vd = 10 mV with non-uniform Vg steps
  pn_diode/
    pn_diode_iv.csv             I-V curves, -2 V ... +0.8 V
  solar_cell/
    solar_cell_jv.csv           J-V curves under illumination, 0 ... 0.77 V
    solar_cell_summary.csv      per cell: Jsc, Voc, Pmax, FF, efficiency (computed from the J-V curve)
  bjt_thermal/
    bjt_thermal_gummel.csv      Gummel plots (V_B sweep at fixed V_C), isothermal and with self-heating
    bjt_thermal_samples.csv     one row per sample: status, number of points, runtime
    bjt_thermal_output.csv      output characteristics I_C(V_C) at fixed base current, isothermal and self-heating
    bjt_thermal_output_samples.csv  one row per sample of the output characteristics
scripts/                        generators, job queue, parser (see "Regenerating the data")
logs/comsol_job_logs.tar.gz     raw COMSOL job logs + Latin hypercube samples of this run (progress, memory, clock-time and machine lines removed; run directory shown as `<RUN>`; `runner.log` times relative to the start)
```

All CSV files have one header row, comma separators and units in the column names.

### MOSFET (2D)
Base model: *DC Characteristics of a MOS Transistor* (Application Library
`Semiconductor_Module/Transistors/mosfet`): n-channel MOSFET, 3 µm × 0.7 µm silicon, p-substrate,
Gaussian n+ source/drain, gate represented by the *Thin Insulator Gate* boundary condition
(εr = 4.5), drift–diffusion with Fermi–Dirac statistics, SRH recombination, Jain–Roulston band gap
narrowing.

* `Id_A_per_um` – drain current in A for **1 µm channel width** (the out-of-plane thickness of the
  2D model).
* Tier 1 device = the Application Library device: NA = 1e17 cm⁻³, tox = 30 nm, Lg = 1.6 µm,
  junction depth 0.25 µm (vertical) / 0.2 µm (lateral), T = 293.15 K.
* Tier 2 parameters (Latin hypercube, random order):

  | Column | Meaning | Range |
  |---|---|---|
  | `NA_cm3` | substrate (channel) acceptor doping | 1e16 – 3e17 cm⁻³ (log) |
  | `tox_nm` | gate oxide thickness | 10 – 40 nm |
  | `Lg_um` | gate length (source/drain implants move with the gate edges) | 0.8 – 2.0 µm |
  | `xj_um` | source/drain junction depth (lateral spread = 0.8 × xj) | 0.15 – 0.35 µm |
  | `T_K` | temperature | 273 – 373 K |

  `sweep` = `idvg` (Vg = 0…5 V in 0.5 V steps at Vd = 10 mV) or `grid` (Vg ∈ {1,2,3,4,5} V ×
  Vd ∈ {0, 0.1, 0.25, 0.5, 1, 2, 3.5, 5} V).
* `mosfet_tier2_idvg_fine.csv`: the same 154 devices at Vd = 10 mV, Vg = 0…5 V in 0.1 V steps within
  ±0.6 V of the analytic long-channel threshold voltage
  V_th = V_FB + 2φ_F + √(2ε_Si·q·N_A·2φ_F)/C_ox, with V_FB = Φ_M − (χ + E_g − kT·ln(N_v/N_A)),
  Φ_M = 4.1 V, χ = 4.05 eV, E_g = 1.12 eV, N_v = 1.04e19·(T/300)^1.5 cm⁻³ (constants of the model),
  and 0.25 V steps elsewhere. This formula agrees with the COMSOL threshold voltage to 41 ± 36 mV.
  The sampling is therefore dense where Id changes fastest (subthreshold → strong inversion).

### pn diode (1D)
Base model: *P–N Junction 1D* (`Semiconductor_Module/Verification_Examples/pn_junction_1d`), finite
volume formulation (Scharfetter–Gummel, the same discretization as DEVSIM). Abrupt silicon junction,
5 µm long, doping-dependent and temperature-dependent mobility, SRH recombination.
Two changes against the Application Library model: the SRH lifetime is set through the model parameter
`taun_param` (the recombination feature takes its lifetime from the material), and the effective
densities of states scale as `Nc, Nv ∝ (T/300 K)^1.5` (the library model keeps nᵢ = 1.25e10 cm⁻³ at every
temperature). A DEVSIM implementation of the same model (notebook `01_pn_diode`) agrees within ~0.1 % in
forward bias.

| Column | Meaning | Range |
|---|---|---|
| `Na_cm3`, `Nd_cm3` | acceptor / donor doping | 1e15 – 1e18 cm⁻³ (log) |
| `tau_s` | electron = hole SRH lifetime | 1e-8 – 1e-5 s (log) |
| `T_K` | temperature | 273 – 373 K |
| `V_V` | applied voltage (positive = forward) | −2 … +0.8 V |
| `I_A` | current for the model's cross-section 4 µm² | |
| `J_A_cm2` | current density = I / 4e-8 cm² | |

### Silicon solar cell (1D)
Base model: *Si Solar Cell 1D* (`Semiconductor_Module/Photonic_Devices_and_Sensors/si_solar_cell_1d`):
n-type base, Gaussian p+ emitter at the illuminated front, generation from the AM1.5 spectrum and the
Si absorption coefficient (250–1300 nm), SRH recombination. Area 1 cm². The SRH lifetime is switched
from the material value (10 µs) to the user-defined value `tau_s`.

| Column | Meaning | Range |
|---|---|---|
| `Nbase_cm3` | n-type base doping | 1e15 – 1e17 cm⁻³ (log) |
| `Nemit_cm3` | p+ emitter peak doping | 1e18 – 1e20 cm⁻³ (log) |
| `xj_um` | emitter junction depth | 0.1 – 1 µm |
| `W_um` | cell thickness | 50 – 300 µm |
| `tau_s` | SRH lifetime | 1e-6 – 1e-3 s (log) |
| `suns` | irradiance relative to AM1.5 (1 sun = 100 mW/cm²) | 0.1 – 1 |
| `T_K` | temperature | 273 – 353 K |
| `V_V` | terminal voltage | 0 … 0.77 V |
| `J_mA_cm2` | current density (positive = generated current) | |

`solar_cell_summary.csv`: `Jsc_mA_cm2` = J(0); `Voc_V` from linear interpolation of the J = 0 crossing;
`Pmax_mW_cm2` = max(V·J) over the sampled voltages (so it slightly underestimates the true maximum);
`FF` = Pmax / (Voc·Jsc); `eta` = Pmax / (100 mW/cm² × suns).

### Bipolar transistor with self-heating (2D)
Base model: *Bipolar Transistor with Self-Heating* (Application Library
`Semiconductor_Module/Transistors/bipolar_transistor_thermal`), **used unchanged** except for the
five sample parameters. npn transistor 2.5 µm wide, out-of-plane length `l_BJT` = 3 µm; emitter
N_E = 2e20, base N_B, n-epi collector N_epi, n+ buried collector N_C = 8e19 cm⁻³; incomplete
ionization; the semiconductor is coupled to *Heat Transfer in Solids* (Joule and recombination heat).
Heat leaves through the emitter, base and collector contacts with the convective condition
q = h0·(T0 − T), where **h0 = 1 / (R · A_contacts)** and A_contacts = (w_cE + w_cB + w_cC) · l_BJT is the
total contact area. `R` (K/W) is therefore the thermal resistance between the device and the ambient:
small R means efficient cooling, large R means poor cooling.

The library model has three chained studies; each sample runs them in the original order:
* Study 5 – incomplete ionization, V_B = 0, V_C ramped to the sample value (initialization, not in the
  data).
* Study 6 → `sweep = isothermal`: Gummel sweep V_B = 0.4, 0.5, 0.6, 0.7, 0.8, 0.85 … 1.1 V at fixed V_C,
  heat transfer off, the device stays at T0.
* Study 7 → `sweep = selfheating`: the same sweep, semiconductor and heat transfer fully coupled.

Batch-mode notes: the Application Library file contains no solutions, and study 1 (the start of the
chain) does not converge from the default initial values in batch mode, although it does in the GUI.
The generator therefore needs the model **saved after computing all studies in the GUI** (see
`setup_models.sh`, `BJT_SOLVED`) and never reruns studies 1–4. Jumping directly from the library
device to other T0 or doping values also diverges, so study 5 is solved as a continuation in one
auxiliary parameter p = 0 … 2 with automatic step control: for p = 0 … 1 T0, N_B and N_epi move from
the library values to the sample values (the doping geometrically); for p = 1 … 2 V_C ramps from 0.5 V
to the sample value. When study 7 does not converge from the V_B = 0 solution (about 1 sample in 6), it
is restarted from the first isothermal point (V_B = 0.4 V), which rescued all but 2 samples. For the
library device the batch run reproduces the stored GUI result (Tmax = 347.51 K at V_B = 1.1 V).

| Column | Meaning | Range |
|---|---|---|
| `T0_K` | ambient temperature (initial and heat sink temperature) | 273 – 343 K |
| `R_K_W` | thermal resistance to the ambient (cooling efficiency, see above; library value 100) | 30 – 300 K/W (log) |
| `VC_V` | collector voltage (emitter grounded; library value 3) | 1 – 5 V |
| `NB_cm3` | base doping (library value 9e17) | 6e17 – 1.4e18 cm⁻³ (log) |
| `Nepi_cm3` | epitaxial collector doping (library value 2e16) | 1e16 – 4e16 cm⁻³ (log) |
| `sweep` | `isothermal` (study 6) or `selfheating` (study 7) | |
| `VB_V` | base voltage | 0.4 … 1.1 V (11 points) |
| `IE_A`, `IB_A`, `IC_A` | terminal currents **into** the contacts (so I_E < 0) for the 3 µm long device | I_C up to 57 mA |
| `Tmax_K` | maximum temperature in the device (= T0 for `isothermal`) | up to 563 K |

Checks done on the data:
* no NaN; Tmax ≥ T0 everywhere;
* at 0.5 V ≤ V_B ≤ 0.8 V the self-heating currents agree with the isothermal ones (median difference
  < 0.5 %);
* at V_B = 1.1 V self-heating raises I_C by 3–56 % (median 14 %) and heats the device by 12–221 K
  (median 58 K); the heating correlates with R · I_C · V_C (r = 0.79);
* all curves are monotonic in V_B from 0.5 V up; I_E + I_B + I_C = 0 to 6e-5 (relative) for
  V_B ≥ 0.6 V and to 5e-3 at V_B = 0.5 V.

**The V_B = 0.4 V points are at the numerical noise floor** (I_C ≈ 1e-11 … 1e-7 A). There
I_E + I_B + I_C deviates from 0 by more than 1 % in 24 % of the isothermal and 51 % of the self-heating
rows, and in 59 self-heating curves I_C(0.4 V) > I_C(0.5 V). Drop V_B = 0.4 V for training (or use it
only with a large current floor).

**Accuracy at high current (cross-check with DEVSIM).** The isothermal Gummel data were
recomputed with DEVSIM (2D, same geometry, doping, and physics – note that the base and buried-collector
Gaussian tails of the library model fall to N_epi, not to the 1e15 cm⁻³ shown in the doping features,
because their background doping is taken from the existing doping). For 9 samples covering the extremes
of all parameters, DEVSIM agrees within 2 % in I_C for V_B ≤ 0.85 V and within 3 % in I_B for
V_B ≤ 0.6 V. At higher current the library mesh (12 000 elements) is too coarse: refining it once
(48 000 elements) lowers the COMSOL I_C of the library device by 6 % at V_B = 1.0 V and by 8 % at 1.1 V,
towards DEVSIM (mesh-converged), and the extrapolation of the two COMSOL meshes lands within 2 % of
DEVSIM. The data are smooth and consistent, so this does not matter for training a surrogate, but
as device physics the high-current points are less accurate: at V_B = 0.95–1.1 V DEVSIM gives 3–28 %
less I_C than COMSOL (most at low T0), and at V_B = 0.7–1.0 V up to 39 % more I_B (at T0 ≈ 273 K). The
mesh refinement was checked only for the library device. In the output characteristics DEVSIM gives
7–15 % less I_C at the same base current (checked for `b0001`).

Ideas for students: learn I_C(V_B, V_C, T0, R, doping) and compare the isothermal and self-heating
predictions; predict Tmax; find the cooling (R) needed to keep Tmax below a limit; look at the current
gain β = I_C / I_B vs. base doping and temperature. Notebook `03_bjt_thermal` does this.

The self-heating is much larger than R · P: a straight-line fit gives (Tmax − T0)/P ≈ 620 K/W + 0.94 · R
at V_B = 1.0 V (r = 0.94). The intercept is the thermal resistance of the silicon itself.

**Output characteristics** (`bjt_thermal_output.csv`): the same 400 samples, common emitter, the base
contact driven by a current. The base contact feature *Base Current* of the library model (used by its
studies 3 and 4, inactive in studies 5–7) is switched on, and V_C is swept from the sample's own V_C
(`VC0_V`; the starting solution is the point of the isothermal Gummel sweep with the closest base
current) down to 0 V and up to 5 V, first with heat transfer off (study `sto1`, `isothermal`) and then
fully coupled, starting from the first isothermal point (study `sto2`, `selfheating`). The studies are
new, with default solvers: the stored solver sequences of studies 3 and 4 belong to the isothermal
library model and do not converge here. The library studies 1–4 themselves cannot be used in batch mode
(they start from T = 0 K, see above).

| Column | Meaning | Range |
|---|---|---|
| `T0_K`, `R_K_W`, `NB_cm3`, `Nepi_cm3` | as in the Gummel data | |
| `VC0_V` | the sample's collector voltage = starting point of the two V_C sweeps (it should not influence the result) | 1 – 5 V |
| `sweep` | `isothermal` or `selfheating` | |
| `IB_A` | applied base current | 10 µA, 100 µA, 1 mA |
| `VC_V` | collector voltage | 0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5 V and `VC0_V` |
| `IE_A`, `IC_A` | emitter and collector currents into the contacts | I_C up to 34 mA |
| `VBE_V` | base voltage needed for `IB_A` | |
| `Tmax_K` | maximum temperature in the device | up to 482 K (heating up to 145 K) |

Checks done on the data:
* no NaN; I_E + I_B + I_C = 0 to 1e-2 (relative) everywhere and to 1e-3 except for 46 self-heating
  points; isothermal Tmax = T0 exactly;
* at V_C = 0 the transistor is in saturation and I_C < 0 in all curves;
* self-heating raises I_C at V_C ≥ 0.5 V by a median of 0.3 % at 10 µA, 1.5 % at 100 µA and 4 % at
  1 mA (up to 22 %); the temperature rise reaches 7 K, 40 K and 145 K;
* I_C is monotonic in V_C from 0.5 V up (one step of 1e-3 relative at 4.99 → 5 V excepted);
* at V_C = 0 the self-heating Tmax can be up to 5 mK *below* T0 (net cooling by recombination and the
  thermoelectric terms), not an error.

A sweep that fails at its first step leaves a single meaningless point in the log (I_E = I_C = 0 or
NaN); the parser drops these.

## Notes for training

* **Currents span many orders of magnitude** (MOSFET subthreshold ~1e-13 A vs. saturation ~1e-4 A;
  diode reverse vs. forward). Train on a transformed target, e.g. `log10(|I| + I_floor)` or
  `asinh(I / I_scale)`, otherwise the network ignores the small currents.
* **Numerical noise floor:** MOSFET currents below ~1e-12 A/µm (and at Vd = 0) are numerical noise
  and can have either sign. Diode current densities below ~1e-8 A/cm² (V ≈ 0, reverse bias of
  long-lifetime diodes) are below the resolution of the simulators: at the heavily doped contacts the
  current is a tiny difference of huge drift and diffusion terms.
* **Solar cell beyond Voc:** points with V > Voc have large negative J (forward diode current). Keep
  them for full J–V learning, or drop rows with `J_mA_cm2 < 0` when learning the power-producing part.
* **Latin hypercube & random order:** any subset of devices is still a reasonable sample. Split
  train/test **by device** (`device_id`), not by row, to test generalization to unseen devices.
* Physics checks for the trained network: MOSFET threshold voltage vs. NA and tox (analytic formula
  in the COMSOL model documentation), diode ideality and saturation current vs. T, solar cell
  Voc temperature coefficient (≈ −2 mV/K) and Jsc ∝ irradiance.

## Loading in Google Colab

```python
import pandas as pd
base = "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/comsol_datasets/data"
mos = pd.read_csv(f"{base}/mosfet/mosfet_tier2.csv")
diode = pd.read_csv(f"{base}/pn_diode/pn_diode_iv.csv")
solar = pd.read_csv(f"{base}/solar_cell/solar_cell_summary.csv")
bjt = pd.read_csv(f"{base}/bjt_thermal/bjt_thermal_gummel.csv")
bjt = bjt[bjt.VB_V >= 0.5]   # V_B = 0.4 V is at the numerical noise floor
bjt_out = pd.read_csv(f"{base}/bjt_thermal/bjt_thermal_output.csv")
```

## Regenerating the data

Requires COMSOL 6.3 with the Semiconductor Module and a bash shell (Git Bash on Windows).
COMSOL batch Java code cannot write files, so each job prints its results to stdout and
`parse_results.sh` collects them from the logs.

0. `setup_models.sh` – copies the base models from your COMSOL Application Library (and the solved
   bipolar transistor model, see above) into the run directory.
1. `prepare_mosfet.sh`, `prepare_diode.sh`, `prepare_solar.sh`, `prepare_bjt.sh` (and
   `prepare_mosfet_fine.sh` for the dense Id–Vg sweeps; `MAKE=make_bjt_out_job.sh SEC=1200 GROUP=71
   TAG=bjto bash prepare_bjt.sh` for the bipolar output characteristics) – sample the device parameters
   (`lhs.awk`, fixed random seeds), write and compile one COMSOL Java class per job
   (`make_*_job.sh`) and put the jobs into the queue directory.
2. `run_queue.sh` – runs up to 4 COMSOL batch processes in parallel (3 cores each), in job-name
   order, with a time limit per job and optional launch deadlines per job group.
3. `parse_results.sh` – builds the CSV files in `data/` from the job logs.

Run directory (logs, queue): `comsol_runs` in your home directory (set `RUN=...` to change it).
