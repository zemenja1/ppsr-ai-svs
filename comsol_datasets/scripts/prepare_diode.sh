#!/usr/bin/env bash
# Builds the 1D pn diode jobs (Latin hypercube over Na, Nd, lifetime, temperature) and queues them.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/pn_junction_1d.mph"
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
N="${N:-500}"
PER_JOB="${PER_JOB:-25}"
mkdir -p "$RUN/build" "$RUN/queue"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/make_diode_job.sh" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

awk -v n="$N" -v seed=1515 -v spec="Na:log:1e15:1e18,Nd:log:1e15:1e18,tau:log:1e-8:1e-5,T:lin:273.15:373.15" \
  -f "$HERE/lhs.awk" > "$RUN/build/diode_lhs.txt"
i=0; j=0; f=""
while read -r na nd tau t; do
  i=$((i + 1))
  if [ $(( (i - 1) % PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "20_diode_$(printf %03d "$j")" 2400 "$f"
    j=$((j + 1)); f="$RUN/build/diode_$j.txt"; : > "$f"
  fi
  echo "d$(printf %04d "$i");$na;$nd;$tau;$t" >> "$f"
done < "$RUN/build/diode_lhs.txt"
job "20_diode_$(printf %03d "$j")" 2400 "$f"
wait
