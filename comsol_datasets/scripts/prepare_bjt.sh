#!/usr/bin/env bash
# Builds the jobs of the bipolar transistor with self-heating (Latin hypercube over ambient temperature,
# package thermal resistance, collector voltage, base and epitaxial doping) and queues them.
#   N=400 PER_JOB=4 GROUP=70 TAG=bjt   (FIRST, COUNT: build only devices FIRST ... FIRST+COUNT-1)
#   MAKE=make_bjt_out_job.sh SEC=1200 GROUP=71 TAG=bjto: output characteristics of the same samples
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/bjt_thermal_lib.mph"   # library model saved with its solutions
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
N="${N:-400}"
PER_JOB="${PER_JOB:-4}"
GROUP="${GROUP:-70}"
TAG="${TAG:-bjt}"
MAKE="${MAKE:-make_bjt_job.sh}"   # job generator
SEC="${SEC:-900}"                 # time limit per sample, s
FIRST="${FIRST:-1}"
COUNT="${COUNT:-$N}"
mkdir -p "$RUN/build" "$RUN/queue"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/$MAKE" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

awk -v n="$N" -v seed=7070 \
  -v spec="T0:lin:273.15:343.15,R:log:30:300,VC:lin:1:5,NB:log:6e17:1.4e18,Nepi:log:1e16:4e16" \
  -f "$HERE/lhs.awk" > "$RUN/build/${TAG}_lhs.txt"   # same seed for every TAG, so the same samples
i=0; j=0; k=0; f=""
while read -r t0 r vc nb nepi; do
  i=$((i + 1))
  [ "$i" -lt "$FIRST" ] || [ "$i" -ge $((FIRST + COUNT)) ] && continue
  if [ $(( k % PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "${GROUP}_${TAG}_$(printf %03d "$j")" $((PER_JOB * SEC)) "$f"
    j=$((j + 1)); f="$RUN/build/${TAG}_$j.txt"; : > "$f"
  fi
  k=$((k + 1))
  echo "b$(printf %04d "$i");$t0;$r;$vc;$nb;$nepi" >> "$f"
done < "$RUN/build/${TAG}_lhs.txt"
job "${GROUP}_${TAG}_$(printf %03d "$j")" $((PER_JOB * SEC)) "$f"
wait
