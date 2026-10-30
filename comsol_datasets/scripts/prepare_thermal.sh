#!/usr/bin/env bash
# Builds the jobs of the 2D diode with self-heating (Latin hypercube over substrate doping, junction depth,
# lifetime, ambient temperature and package thermal resistance) and queues them.
#   N=240 PER_JOB=6 GROUP=60 TAG=thermal   (FIRST, COUNT: build only devices FIRST ... FIRST+COUNT-1)
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/mosfet_base.mph"
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
N="${N:-240}"
PER_JOB="${PER_JOB:-6}"
GROUP="${GROUP:-60}"
TAG="${TAG:-thermal}"
FIRST="${FIRST:-1}"
COUNT="${COUNT:-$N}"
mkdir -p "$RUN/build" "$RUN/queue"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/make_thermal_job.sh" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

awk -v n="$N" -v seed=6060 \
  -v spec="NA:log:1e16:3e17,xj:lin:0.15:0.35,tau:log:1e-8:1e-5,T0:lin:273.15:373.15,rth:log:0.01:0.1" \
  -f "$HERE/lhs.awk" > "$RUN/build/${TAG}_lhs.txt"
i=0; j=0; k=0; f=""
while read -r na xj tau t0 rth; do
  i=$((i + 1))
  [ "$i" -lt "$FIRST" ] || [ "$i" -ge $((FIRST + COUNT)) ] && continue
  if [ $(( k % PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "${GROUP}_${TAG}_$(printf %03d "$j")" $((PER_JOB * 900)) "$f"
    j=$((j + 1)); f="$RUN/build/${TAG}_$j.txt"; : > "$f"
  fi
  k=$((k + 1))
  echo "h$(printf %04d "$i");$na;$xj;$tau;$t0;$rth" >> "$f"
done < "$RUN/build/${TAG}_lhs.txt"
job "${GROUP}_${TAG}_$(printf %03d "$j")" $((PER_JOB * 900)) "$f"
wait
