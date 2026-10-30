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
| 4. Diode with self-heating (2D) | built from `mosfet` | NA, junction depth, lifetime, ambient temperature, thermal resistance, bias | V, I, junction temperature Tj |

The models are from the COMSOL Application Library (Semiconductor Module). The `.mph` files are
licensed under the COMSOL Software License Agreement and are **not** included in this repository;
`scripts/setup_models.sh` copies them from your own COMSOL installation.

## Contents of the current version (v4, generated 2026-10-29/30, COMSOL 6.3, ~20 h on a laptop in total)

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
| `thermal_diode/thermal_diode_iv.csv` | 7 520 | 188 diodes × (23 isothermal + 17 self-heating points) |
| `thermal_diode/thermal_diode_devices.csv` | 210 | 188 `ok`, 22 `failed` (see below) |

Known gaps:
* MOSFET devices `m0055` and `m0067` (NA ≈ 2.6e17 cm⁻³ at T ≈ 282–286 K) did not converge at the first
  bias point and have no data, so the corner *high doping + low temperature* is slightly under-sampled.
* Tier 2 consists of the 150-device Latin hypercube (`m0001`–`m0150`) plus the first 6 devices of a
  second hypercube (`m0151`–`m0156`); the second batch was cut by the time budget.
* The solar cells are the first 450 of a 600-cell Latin hypercube (cut by the time budget; because
  the sample order is random, the 450 cells still cover the whole parameter space).
* Thermal diode: 210 of a 300-diode Latin hypercube were computed (cut by the time budget). 22 diodes
  failed in the isothermal sweep and have no data; 19 of them have NA < 2.2e16 cm⁻³ (19 of the 50
  diodes in that range failed), so **low substrate doping is under-sampled** – keep this in mind when
  testing the network near NA = 1e16 cm⁻³.

### Version history
* **v1 (2026-10-29, morning):** in the pn diode and solar cell data the `tau_s` column had **no
  effect** – the COMSOL recombination features took the lifetime from the material, so the value set by
  the generator was ignored (diode: τ = 1 µs, solar cell: τ = 10 µs for all devices). In the diode data
  nᵢ was also independent of temperature, so currents *decreased* with T. Found when comparing COMSOL
  with DEVSIM; both datasets were regenerated.
* **v2 (2026-10-29, late morning):** lifetime and nᵢ(T) corrected. But in ~60 % of the diode curves one
  reverse-bias point (and in ~1 % of the solar J–V curves one low-voltage point) was wrong: COMSOL's
  default convergence test checks only the size of the Newton update, and accepted iterations that had
  stalled with a large residual – without any warning. Found by checking the curves for smoothness.
* **v3 (2026-10-29):** pn diode and solar cell data regenerated with `termonres = on` (the residual must
  be small, too) for all nonlinear solvers. Reverse currents now agree with DEVSIM to ~4 digits above
  ~1e-7 A/cm². All 500 diode I–V curves are monotonic. In 13 solar J–V curves J rises by < 3e-5
  (relative) between neighbouring points on the flat plateau below the maximum power point – solver
  tolerance, not an error. The MOSFET data were checked (all Id(Vd) and Id(Vg) curves monotonic) and
  not regenerated.
* **v4 (2026-10-29/30, night):** added `mosfet_tier2_idvg_fine.csv` (dense Id–Vg sweeps of all tier 2
  devices, agree with the 0.5 V points of `mosfet_tier2.csv` to 2e-5 relative) and the thermal diode
  dataset. Existing files unchanged.

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
  thermal_diode/
    thermal_diode_iv.csv        isothermal I-V and self-heating V-I curves with junction temperature
    thermal_diode_devices.csv   one row per diode: status, number of points, runtime
scripts/                        generators, job queue, parser (see "Regenerating the data")
logs/comsol_job_logs.tar.gz     raw COMSOL job logs + Latin hypercube samples of this run
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

### Diode with self-heating (2D)
Built from the MOSFET model above (the Application Library has no stand-alone power diode; the
*bipolar_transistor_thermal* model does not converge in batch mode): the gate contact is removed, both
n+ regions form the cathode (0 V), and the substrate contact at the bottom (3 µm wide) is the anode
and the heat sink. Chip area A = 3 µm × 1 µm = 3e-8 cm². Mobilities depend on temperature
(μn = 1448·(T/300)^−2.33, μp = 473·(T/300)^−2.23 cm²/Vs), nᵢ(T) through Nc, Nv ∝ T^1.5, SRH lifetime
`tau_s` for electrons and holes.

**Thermal model (lumped).** The temperature drop inside 0.7 µm of silicon is < 0.02 K, so the chip is
isothermal at the junction temperature Tj, and the heat flows to the ambient through the package
thermal resistance r_th (per unit area):

    Tj = T0 + r_th · V · J        (global equation solved together with the semiconductor equations)

A fully coupled 2D heat transfer model gives the same result but converged far too slowly.

**Two sweeps per diode:**
* `isothermal` – voltage driven, V = 0…0.9 V (23 points), Tj = T0 (no self-heating).
* `selfheating` – **current driven**, J = 0.1 … 1000 A/cm² (17 points, 4 per decade). Under voltage
  drive the diode runs away thermally (d ln J / dT ≈ 0.15 /K, so a few K of heating already doubles
  the current) and the solver fails near V ≈ 0.7 V; with current drive the operating point is stable.
  Typical heating at 1000 A/cm² is 21 K (max 68 K). In 55 diodes the voltage *decreases* above
  ~300 A/cm² – negative differential resistance caused by self-heating, physically correct.

| Column | Meaning | Range |
|---|---|---|
| `NA_cm3` | substrate (anode) acceptor doping | 1e16 – 3e17 cm⁻³ (log) |
| `xj_um` | n+ junction depth (lateral spread 0.8 × xj) | 0.15 – 0.35 µm |
| `tau_s` | SRH lifetime | 1e-8 – 1e-5 s (log) |
| `T0_K` | ambient (heat sink) temperature | 273 – 373 K |
| `rth_Kcm2_W` | area-specific thermal resistance | 0.01 – 0.1 K·cm²/W (log) |
| `sweep` | `isothermal` or `selfheating` | |
| `V_V` | anode voltage | 0 … 0.9 V (isothermal) |
| `I_A` | anode current | |
| `J_A_cm2` | current density = I / 3e-8 cm² | 0.1 … 1000 A/cm² (self-heating) |
| `Tj_K` | junction temperature (= T0 for `isothermal`) | up to 421 K |

Checks done on the data: the energy balance Tj − T0 = r_th·V·J holds to 5e-5 K; at 0.1 A/cm² the
self-heating voltage agrees with the isothermal curve to 0.01 mV (median, max 0.11 mV); all
self-heating sweeps are complete (17 points). Isothermal currents below ~1e-14 A (V ≤ 0.1 V) are at the
numerical noise floor and can be non-monotonic or negative. Note: the solver for the self-heating
sweep runs without the residual test (the scaled residual of the coupled system stalls above the
tolerance), which is why the checks above were done afterwards.

Ideas for students: learn V(J, T0, r_th, …) and compare with the isothermal curve to see the
self-heating; predict Tj; check that the network reproduces the energy balance and the temperature
coefficient of the forward voltage (≈ −2 mV/K at constant current).

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
thermal = pd.read_csv(f"{base}/thermal_diode/thermal_diode_iv.csv")
```

## Regenerating the data

Requires COMSOL 6.3 with the Semiconductor Module and a bash shell (Git Bash on Windows).
COMSOL batch Java code cannot write files, so each job prints its results to stdout and
`parse_results.sh` collects them from the logs.

0. `setup_models.sh` – copies the three base models from your COMSOL Application Library into the
   run directory.
1. `prepare_mosfet.sh`, `prepare_diode.sh`, `prepare_solar.sh`, `prepare_thermal.sh` (and
   `prepare_mosfet_fine.sh` for the dense Id–Vg sweeps) – sample the device parameters
   (`lhs.awk`, fixed random seeds), write and compile one COMSOL Java class per job
   (`make_*_job.sh`) and put the jobs into the queue directory.
2. `run_queue.sh` – runs up to 4 COMSOL batch processes in parallel (3 cores each), in job-name
   order, with a time limit per job and optional launch deadlines per job group.
3. `parse_results.sh` – builds the CSV files in `data/` from the job logs.

Run directory (logs, queue): `C:/Users/janze/comsol_runs` (set `RUN=...` to change it).
