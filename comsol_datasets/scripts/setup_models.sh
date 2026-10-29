#!/usr/bin/env bash
# Copies the base models from your own COMSOL Application Library into the run directory.
# (The .mph files are licensed under the COMSOL Software License Agreement and are therefore
# not part of this repository.)
RUN="${RUN:-C:/Users/janze/comsol_runs}"
LIB="${LIB:-C:/Program Files/COMSOL/COMSOL63/Multiphysics/applications/Semiconductor_Module}"
mkdir -p "$RUN/models"
cp "$LIB/Transistors/mosfet.mph"                         "$RUN/models/mosfet_base.mph"
cp "$LIB/Verification_Examples/pn_junction_1d.mph"       "$RUN/models/pn_junction_1d.mph"
cp "$LIB/Photonic_Devices_and_Sensors/si_solar_cell_1d.mph" "$RUN/models/si_solar_cell_1d.mph"
ls -l "$RUN/models"
