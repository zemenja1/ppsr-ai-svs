#!/usr/bin/env bash
# Copies the base models from your own COMSOL Application Library into the run directory.
# (The .mph files are licensed under the COMSOL Software License Agreement and are therefore
# not part of this repository.)
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
LIB="${LIB:-C:/Program Files/COMSOL/COMSOL63/Multiphysics/applications/Semiconductor_Module}"
mkdir -p "$RUN/models"
cp "$LIB/Transistors/mosfet.mph"                         "$RUN/models/mosfet_base.mph"
cp "$LIB/Verification_Examples/pn_junction_1d.mph"       "$RUN/models/pn_junction_1d.mph"
cp "$LIB/Photonic_Devices_and_Sensors/si_solar_cell_1d.mph" "$RUN/models/si_solar_cell_1d.mph"
# bipolar_transistor_thermal: the library file has no stored solutions, and its study 1 does not converge from
# the default initial values in batch mode. Open the model in the COMSOL GUI, compute all studies, save it
# (~56 MB) and point BJT_SOLVED to that file.
BJT_SOLVED="${BJT_SOLVED:-$HERE/../../../Comsol_models/bipolar_transistor_thermal.mph}"
[ -f "$BJT_SOLVED" ] && cp "$BJT_SOLVED" "$RUN/models/bjt_thermal_lib.mph" || echo "skipped bjt_thermal_lib.mph (set BJT_SOLVED)"
ls -l "$RUN/models"
