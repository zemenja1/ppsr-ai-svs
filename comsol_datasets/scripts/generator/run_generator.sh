#!/usr/bin/env bash
# Generator yoke sweep: permeability scaling of the iron B(H) curve x conductivity and lamination thickness
# of the Bertotti loss model. Needs COMSOL 6.3 with the AC/DC Module (about 30 min for the full sweep).
# Usage: bash run_generator.sh [kmu list]   (default: 1 plus 20 values from 0.01 to 3, log spaced)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-$(cygpath -m "$HOME" 2>/dev/null || echo "$HOME")/comsol_runs}"
BIN="${BIN:-C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64}"
KMU="${1:-1.0, 0.01, 0.0135, 0.01823, 0.02461, 0.03323, 0.04486, 0.06057, 0.08178, 0.1104, 0.1491, 0.2013, 0.2717, 0.3669, 0.4953, 0.6687, 0.9029, 1.219, 1.646, 2.222, 3}"
W="$RUN/generator"; mkdir -p "$W"
MPH="${GEN_MPH:-$RUN/models/generator_2d.mph}"
sed -e "s/%KMU%/$KMU/" -e "s#%MPH%#$MPH#" "$HERE/GenRun.java" > "$W/GenRun.java"
"$BIN/comsolcompile.exe" "$W/GenRun.java"
"$BIN/comsolbatch.exe" -inputfile "$(cygpath -m "$W/GenRun.class" 2>/dev/null || echo "$W/GenRun.class")" -np 4 -nosave > "$W/generator.log" 2>&1
OUT="${OUT:-$HERE/../../data/generator_2d}"
python "$HERE/convert_generator.py" "$W/generator.log" "$OUT"
python "$HERE/plot_generator.py" "$OUT"
