# ppsr-ai-svs
Training DNN to model semiconductor devices.

Teaching materials of the project *AI ve výuce – Elektrotechnologie* (Czech Technical University in
Prague, Faculty of Electrical Engineering). Students train deep neural networks as fast surrogate
models of semiconductor device simulations and compare them with COMSOL Multiphysics' built-in DNN
surrogates.

## Case studies

| # | Device | Data | Training |
|---|---|---|---|
| 1 | pn diode (1D) | COMSOL dataset (teacher) + students' own DEVSIM simulations | I–V curves from doping, lifetime, temperature |
| 2 | Silicon solar cell (1D) | COMSOL dataset | J–V curves and Jsc, Voc, FF, efficiency from cell design and operating conditions |
| 3 | MOSFET (2D) | COMSOL dataset | Id(Vg, Vd) of one device (tier 1) and of many device designs (tier 2) |

## Repository layout

```
comsol_datasets/   COMSOL training data (CSV), generation scripts, documentation
notebooks/         Google Colab notebooks (coming)
```

See [comsol_datasets/README.md](comsol_datasets/README.md) for the description of all datasets.

## Using the data in Google Colab

```python
import pandas as pd
base = "https://raw.githubusercontent.com/zemenja1/ppsr-ai-svs/main/comsol_datasets/data"
diode = pd.read_csv(f"{base}/pn_diode/pn_diode_iv.csv")
solar = pd.read_csv(f"{base}/solar_cell/solar_cell_summary.csv")
mosfet = pd.read_csv(f"{base}/mosfet/mosfet_tier2.csv")
```

## License

* Source code (scripts, notebook code): [MIT](LICENSE)
* Datasets and teaching materials: [CC BY 4.0](LICENSE-DATA)

COMSOL model files (`.mph`) are not included: they are licensed under the COMSOL Software License
Agreement. The scripts copy them from your own COMSOL installation
(`comsol_datasets/scripts/setup_models.sh`).
