#!/usr/bin/env bash
# Builds the MOSFET tier 1 and tier 2 jobs and puts them into the queue.
# Device line format (see make_mosfet_job.sh):
#   tier;id;NA_cm3;tox_nm;Lg_um;xj_um;T_K;Vg list for Id-Vg (Vd=10 mV);Vg list for grid;Vd list for grid

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/mosfet_base.mph"
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
N_T2="${N_T2:-150}"
T2_SEED="${T2_SEED:-2026}"        # seed of the Latin hypercube
T2_FIRST_ID="${T2_FIRST_ID:-1}"   # number of the first device id (m0001, ...)
T2_GROUP="${T2_GROUP:-30}"        # job group (queue priority / deadline)
T2_TAG="${T2_TAG:-t2}"            # job name part, e.g. t2 or t2b for an extra batch
DEVS_PER_JOB="${DEVS_PER_JOB:-3}"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/make_mosfet_job.sh" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

mkdir -p "$RUN/build" "$RUN/queue"
REF="1e17;30;1.6;0.25;293.15"   # COMSOL Application Library reference device

# Tier 1: reference device, dense grid split over 4 jobs by Vg, plus a fine Id-Vg curve.
if [ -z "$SKIP_T1" ]; then
VG_T1=$(seq 0 0.25 5 | tr '\n' ' ')
VD_T1="0 0.05 0.1 0.25 0.5 0.75 1 1.25 1.5 2 2.5 3 3.5 4 4.5 5"
k=0
for chunk in "0 0.25 0.5 0.75 1 1.25" "1.5 1.75 2 2.25 2.5" "2.75 3 3.25 3.5 3.75 4" "4.25 4.5 4.75 5"; do
  k=$((k + 1))
  f="$RUN/build/t1_$k.txt"
  echo "t1;ref;$REF;$VG_T1;$chunk;$VD_T1" > "$f"
  job "10_mos_t1_grid$k" 3600 "$f"
done
f="$RUN/build/t1_idvg.txt"
echo "t1;ref;$REF;$(seq 0 0.1 5 | tr '\n' ' ');0;0 0.01" > "$f"
job "11_mos_t1_idvg" 2400 "$f"
fi

# Tier 2: Latin hypercube over device parameters, in random order so any prefix is usable.
VG1_T2="0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5"
VG2_T2="1 2 3 4 5"
VD_T2="0 0.1 0.25 0.5 1 2 3.5 5"
awk -v n="$N_T2" -v seed="$T2_SEED" \
  -v spec="NA:log:1e16:3e17,tox:lin:10:40,Lg:lin:0.8:2.0,xj:lin:0.15:0.35,T:lin:273.15:373.15" \
  -f "$HERE/lhs.awk" > "$RUN/build/${T2_TAG}_lhs.txt"
i=0; j=0; f=""
while read -r na tox lg xj t; do
  i=$((i + 1))
  if [ $(( (i - 1) % DEVS_PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "${T2_GROUP}_mos_${T2_TAG}_$(printf %03d "$j")" 4500 "$f"
    j=$((j + 1)); f="$RUN/build/${T2_TAG}_$j.txt"; : > "$f"
  fi
  echo "t2;m$(printf %04d $((i + T2_FIRST_ID - 1)));$na;$tox;$lg;$xj;$t;$VG1_T2;$VG2_T2;$VD_T2" >> "$f"
done < "$RUN/build/${T2_TAG}_lhs.txt"
job "${T2_GROUP}_mos_${T2_TAG}_$(printf %03d "$j")" 4500 "$f"
wait
