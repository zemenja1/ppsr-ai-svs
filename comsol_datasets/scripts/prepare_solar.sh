#!/usr/bin/env bash
# Builds the 1D solar cell jobs (Latin hypercube over doping, emitter depth, thickness, lifetime,
# irradiance and temperature) and queues them.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/si_solar_cell_1d.mph"
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
N="${N:-600}"
PER_JOB="${PER_JOB:-30}"
mkdir -p "$RUN/build" "$RUN/queue"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/make_solar_job.sh" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

awk -v n="$N" -v seed=4242 -v spec="Nb:log:1e15:1e17,Ne:log:1e18:1e20,xj:lin:0.1:1,W:lin:50:300,tau:log:1e-6:1e-3,suns:lin:0.1:1,T:lin:273.15:353.15" \
  -f "$HERE/lhs.awk" > "$RUN/build/solar_lhs.txt"
i=0; j=0; f=""
while read -r nb ne xj w tau suns t; do
  i=$((i + 1))
  if [ $(( (i - 1) % PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "40_solar_$(printf %03d "$j")" 2700 "$f"
    j=$((j + 1)); f="$RUN/build/solar_$j.txt"; : > "$f"
  fi
  echo "s$(printf %04d "$i");$nb;$ne;$xj;$w;$tau;$suns;$t" >> "$f"
done < "$RUN/build/solar_lhs.txt"
job "40_solar_$(printf %03d "$j")" 2700 "$f"
wait
