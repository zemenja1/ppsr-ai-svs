#!/usr/bin/env bash
# Builds jobs with dense Id-Vg sweeps for the tier 2 MOSFET devices and puts them into the queue
# (generator: make_mosfet_fine_job.sh). The Vg list of each device is non-uniform: DENSE V steps within
# +-0.6 V of the analytic long-channel threshold voltage, 0.25 V steps elsewhere (0 ... 5 V).
#   DEVICES="m0001 m0055"  only these devices (default: all devices in mosfet_tier2_devices.csv that
#                          converged in the first run)
#   DENSE=0.1              step near the threshold voltage (a multiple of 0.05 V)

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
BASE="$RUN/models/mosfet_base.mph"
COMPILE="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"
DEVFILE="${DEVFILE:-$HERE/../data/mosfet/mosfet_tier2_devices.csv}"
GROUP="${GROUP:-50}"
TAG="${TAG:-fine}"
DEVS_PER_JOB="${DEVS_PER_JOB:-4}"
VD_RAMP="${VD_RAMP-0.01 0.05 0.1 0.25 0.5 1 2}"    # last value = drain voltage of the second Id-Vg sweep;
                                                    # VD_RAMP="" computes only the sweep at Vd = 10 mV
DENSE="${DENSE:-0.1}"

job() {  # job <name> <limit_s> <device file>; compiles up to 4 jobs in parallel
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
  (
    local name="$1" dir="$RUN/build/$1" cls
    cls="J$(echo "$name" | tr -cd 'A-Za-z0-9')"
    mkdir -p "$dir"
    bash "$HERE/make_mosfet_fine_job.sh" "$cls" "$3" "$BASE" "$dir/$cls.java"
    (cd "$dir" && "$COMPILE" "$cls.java" > compile.log 2>&1)
    [ -f "$dir/$cls.class" ] || { echo "compile failed: $name"; exit 1; }
    echo "$2" > "$dir/limit"
    cp "$3" "$dir/devices.txt"
    mv "$dir" "$RUN/queue/$name"
    echo "queued $name"
  ) &
}

mkdir -p "$RUN/build" "$RUN/queue"
# device lines: id;NA;tox;Lg;xj;T;Vg list;Vd ramp
# Devices that failed in the first run have no parameters in the CSV; they come from the saved sample.
awk -F, -v sel=" $DEVICES " -v vd="$VD_RAMP" -v lhs="$RUN/build/t2_lhs.txt" -v dk="$(awk -v d="$DENSE" 'BEGIN { print int(d / 0.05 + 0.5) }')" '
  function vth(NA, tox, T,   kT, Nc, Nv, ni, p2, vfb, cox) {   # same formula as in notebook 02_mosfet
    kT = 8.617333e-5 * T; Nc = 2.8e19 * (T / 300) ^ 1.5; Nv = 1.04e19 * (T / 300) ^ 1.5
    ni = sqrt(Nc * Nv) * exp(-1.12 / (2 * kT)); p2 = 2 * kT * log(NA / ni)
    vfb = 4.1 - (4.05 + 1.12 - kT * log(Nv / NA)); cox = 4.5 * 8.854188e-14 / (tox * 1e-7)
    return vfb + p2 + sqrt(2 * 11.7 * 8.854188e-14 * 1.602177e-19 * NA * p2) / cox
  }
  NR > 1 && ((sel == "  " && $7 == "ok") || index(sel, " " $1 " ")) {
    if ($2 == "") {
      n = substr($1, 2) + 0; k = 0
      while ((getline l < lhs) > 0) if (++k == n) { split(l, p, " "); $2 = p[1]; $3 = p[2]; $4 = p[3]; $5 = p[4]; $6 = p[5] }
      close(lhs)
    }
    v = vth($2, $3, $6); lo = int((v - 0.6) / 0.05 + 0.5); hi = int((v + 0.6) / 0.05 + 0.5)
    list = ""
    for (k = 0; k <= 100; k++)   # k = Vg / 0.05 V
      if ((k >= lo && k <= hi && k % dk == 0) || k % 5 == 0) list = list sprintf("%g ", k * 0.05)
    printf "%s;%s;%s;%s;%s;%s;%s;%s\n", $1, $2, $3, $4, $5, $6, list, vd
  }' "$DEVFILE" > "$RUN/build/${TAG}_devices.txt"
echo "$(wc -l < "$RUN/build/${TAG}_devices.txt") devices"

i=0; j=0; f=""
while read -r line; do
  i=$((i + 1))
  if [ $(( (i - 1) % DEVS_PER_JOB )) -eq 0 ]; then
    [ -n "$f" ] && job "${GROUP}_mos_${TAG}_$(printf %03d "$j")" $((DEVS_PER_JOB * 900)) "$f"
    j=$((j + 1)); f="$RUN/build/${TAG}_$j.txt"; : > "$f"
  fi
  echo "$line" >> "$f"
done < "$RUN/build/${TAG}_devices.txt"
job "${GROUP}_mos_${TAG}_$(printf %03d "$j")" $((DEVS_PER_JOB * 900)) "$f"
wait
