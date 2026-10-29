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

The models are from the COMSOL Application Library (Semiconductor Module). The `.mph` files are
licensed under the COMSOL Software License Agreement and are **not** included in this repository;
`scripts/setup_models.sh` copies them from your own COMSOL installation.

## Contents of the current version (v3, generated 2026-10-29, COMSOL 6.3, ~8 h on a laptop in total)

| File | Rows | Contents |
|---|---|---|
| `mosfet/mosfet_tier1_grid.csv` | 336 | 21 Vg × 16 Vd |
| `mosfet/mosfet_tier1_idvg.csv` | 51 | Vg = 0…5 V, step 0.1 V |
| `mosfet/mosfet_tier2.csv` | 7 854 | 154 devices × (11 Id–Vg + 40 grid points) |
| `mosfet/mosfet_tier2_devices.csv` | 156 | 154 `ok`, 2 `failed` (see below) |
| `pn_diode/pn_diode_iv.csv` | 11 500 | 500 diodes × 23 voltages |
| `solar_cell/solar_cell_jv.csv` | 11 700 | 450 cells × 26 voltages |
| `solar_cell/solar_cell_summary.csv` | 450 | one row per cell |

Known gaps:
* MOSFET devices `m0055` and `m0067` (NA ≈ 2.6e17 cm⁻³ at T ≈ 282–286 K) did not converge at the first
  bias point and have no data, so the corner *high doping + low temperature* is slightly under-sampled.
* Tier 2 consists of the 150-device Latin hypercube (`m0001`–`m0150`) plus the first 6 devices of a
  second hypercube (`m0151`–`m0156`); the second batch was cut by the time budget.
* The solar cells are the first 450 of a 600-cell Latin hypercube (cut by the time budget; because
  the sample order is random, the 450 cells still cover the whole parameter space).

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

## Files

```
data/
  mosfet/
    mosfet_tier1_grid.csv       reference device, dense Vg x Vd grid
    mosfet_tier1_idvg.csv       reference device, Id-Vg at Vd = 10 mV (threshold region, fine Vg steps)
    mosfet_tier2.csv            many devices (Latin hypercube), Id-Vg at Vd = 10 mV + Vg x Vd grid
    mosfet_tier2_devices.csv    one row per device: parameters, status, number of points, runtime
  pn_diode/
    pn_diode_iv.csv             I-V curves, -2 V ... +0.8 V
  solar_cell/
    solar_cell_jv.csv           J-V curves under illumination, 0 ... 0.77 V
    solar_cell_summary.csv      per cell: Jsc, Voc, Pmax, FF, efficiency (computed from the J-V curve)
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
```

## Regenerating the data

Requires COMSOL 6.3 with the Semiconductor Module and a bash shell (Git Bash on Windows).
COMSOL batch Java code cannot write files, so each job prints its results to stdout and
`parse_results.sh` collects them from the logs.

0. `setup_models.sh` – copies the three base models from your COMSOL Application Library into the
   run directory.
1. `prepare_mosfet.sh`, `prepare_diode.sh`, `prepare_solar.sh` – sample the device parameters
   (`lhs.awk`, fixed random seeds), write and compile one COMSOL Java class per job
   (`make_*_job.sh`) and put the jobs into the queue directory.
2. `run_queue.sh` – runs up to 4 COMSOL batch processes in parallel (3 cores each), in job-name
   order, with a time limit per job and optional launch deadlines per job group.
3. `parse_results.sh` – builds the CSV files in `data/` from the job logs.

Run directory (logs, queue): `C:/Users/janze/comsol_runs` (set `RUN=...` to change it).
