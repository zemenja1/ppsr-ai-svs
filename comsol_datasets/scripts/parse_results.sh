#!/usr/bin/env bash
# Collects the COMSOL job logs into CSV datasets (one header row, comma separated, SI-style units in
# the column names). Safe to run repeatedly while jobs are still running; duplicate rows are dropped.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-C:/Users/janze/comsol_runs}"
OUT="${OUT:-$HERE/../data}"
LOGS="$RUN/logs"
mkdir -p "$OUT/mosfet" "$OUT/pn_diode" "$OUT/solar_cell"
shopt -s nullglob

# ---------- MOSFET (2D, out-of-plane width 1 um, so Id is in A per um of channel width) ----------
t1=("$LOGS"/10_mos_t1_*.log); t1i=("$LOGS"/11_mos_t1_*.log); t2=("$LOGS"/3?_mos_t2*.log)
{
  echo "Vg_V,Vd_V,Id_A_per_um"
  [ ${#t1[@]} -gt 0 ] && awk -F, '$1=="GRID" && $2=="t1" { k=$9","$10; if (!(k in s)) { s[k]=1; print $9","$10","$11 } }' "${t1[@]}" | sort -t, -k1,1g -k2,2g
} > "$OUT/mosfet/mosfet_tier1_grid.csv"
{
  echo "Vg_V,Vd_V,Id_A_per_um"
  [ ${#t1i[@]} -gt 0 ] && awk -F, '$1=="IDVG" && $2=="t1" { k=$9; if (!(k in s)) { s[k]=1; print $9","$10","$11 } }' "${t1i[@]}" | sort -t, -k1,1g
} > "$OUT/mosfet/mosfet_tier1_idvg.csv"
{
  echo "device_id,NA_cm3,tox_nm,Lg_um,xj_um,T_K,Vg_V,Vd_V,Id_A_per_um,sweep"
  [ ${#t2[@]} -gt 0 ] && awk -F, '($1=="GRID" || $1=="IDVG") && $2=="t2" {
      sw = ($1 == "IDVG") ? "idvg" : "grid"; k = $3","$9","$10","sw
      if (!(k in s)) { s[k]=1; print $3","$4","$5","$6","$7","$8","$9","$10","$11","sw } }' "${t2[@]}" | sort -t, -k1,1 -k10,10 -k7,7g -k8,8g
} > "$OUT/mosfet/mosfet_tier2.csv"
{
  echo "device_id,NA_cm3,tox_nm,Lg_um,xj_um,T_K,status,n_idvg,n_grid,runtime_s"
  [ ${#t2[@]} -gt 0 ] && awk -F, '
    $1=="GRID" && $2=="t2" { p[$3] = $4","$5","$6","$7","$8 }
    $1=="IDVG" && $2=="t2" { p[$3] = $4","$5","$6","$7","$8 }
    $1=="DONE" { st[$2]="ok"; n1[$2]=$3; n2[$2]=$4; rt[$2]=$5 }
    $1=="FAIL" { if (!($2 in st)) st[$2]="failed" }
    $1=="FAILVG" { fv[$2]++ }
    END { for (d in st) if (d in p || st[d]=="failed") {
            s = st[d]; if (s == "ok" && fv[d] > 0) s = "partial"
            print d","(d in p ? p[d] : ",,,,")","s","n1[d]","n2[d]","rt[d] } }' "${t2[@]}" | sort
} > "$OUT/mosfet/mosfet_tier2_devices.csv"
# Dense Id-Vg sweeps (Vd = 10 mV) of the tier 2 devices, non-uniform Vg steps around the threshold voltage
tf=("$LOGS"/50_mos_fine_*.log)
{
  echo "device_id,NA_cm3,tox_nm,Lg_um,xj_um,T_K,Vg_V,Vd_V,Id_A_per_um"
  [ ${#tf[@]} -gt 0 ] && awk -F, '$1=="FINE" && $2=="t2" { k=$3","$9; if (!(k in s)) { s[k]=1; print $3","$4","$5","$6","$7","$8","$9","$10","$11 } }' "${tf[@]}" | sort -t, -k1,1 -k7,7g
} > "$OUT/mosfet/mosfet_tier2_idvg_fine.csv"

# ---------- 2D diode with self-heating (chip area 3 um x 1 um = 3e-8 cm^2) ----------
mkdir -p "$OUT/thermal_diode"
th=("$LOGS"/60_thermal_*.log)
{
  echo "device_id,NA_cm3,xj_um,tau_s,T0_K,rth_Kcm2_W,sweep,V_V,I_A,J_A_cm2,Tj_K"
  # isothermal rows ordered by V (the current near V = 0 is numerical noise), self-heating rows by J
  [ ${#th[@]} -gt 0 ] && awk -F, '($1=="ISO" || $1=="HOT") {
      sw = ($1 == "ISO") ? "isothermal" : "selfheating"; k = $2","sw","$8","$9
      if (!(k in s)) { s[k]=1; printf "%.9e,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.6e,%s\n", ($1 == "ISO") ? $8 : $9, $2,$3,$4,$5,$6,$7,sw,$8,$9,$9/3e-8,$10 } }' "${th[@]}" |
    sort -t, -k2,2 -k8,8 -k1,1g | cut -d, -f2-
} > "$OUT/thermal_diode/thermal_diode_iv.csv"
{
  echo "device_id,status,n_isothermal,n_selfheating,runtime_s"
  [ ${#th[@]} -gt 0 ] && awk -F, '
    $1=="DONE" { st[$2] = ($4 == 17) ? "ok" : "partial"; n1[$2]=$3; n2[$2]=$4; rt[$2]=$5 }
    $1=="FAIL" { if (!($2 in st)) st[$2]="failed" }
    END { for (d in st) print d","st[d]","n1[d]","n2[d]","rt[d] }' "${th[@]}" | sort
} > "$OUT/thermal_diode/thermal_diode_devices.csv"

# ---------- bipolar transistor with self-heating (2D, out-of-plane length 3 um; currents into the contacts) ----------
mkdir -p "$OUT/bjt_thermal"
bj=("$LOGS"/70_bjt*.log)
# Only the 11 V_B points of the Gummel sweep are kept: a study that stops early leaves its last step-reduced
# point (off grid) or, if std7 never converged, the stored V_B = 0 solution in the dataset.
BJGRID='function ongrid(v) { v += 0; return v > 0.39 && v < 1.11 && (v * 20 - int(v * 20 + 0.5)) ^ 2 < 1e-8 }'
{
  echo "sample_id,T0_K,R_K_W,VC_V,NB_cm3,Nepi_cm3,sweep,VB_V,IE_A,IB_A,IC_A,Tmax_K"
  [ ${#bj[@]} -gt 0 ] && awk -F, "$BJGRID"'
    ($1=="ISO" || $1=="HOT") && ongrid($8) {
      sw = ($1 == "ISO") ? "isothermal" : "selfheating"; k = $2","sw","$8
      if (!(k in s)) { s[k]=1; print $2","$3","$4","$5","$6","$7","sw","$8","$9","$10","$11","$12 } }' "${bj[@]}" |
    sort -t, -k1,1 -k7,7 -k8,8g
} > "$OUT/bjt_thermal/bjt_thermal_gummel.csv"
{
  echo "sample_id,status,n_isothermal,n_selfheating,runtime_s"
  [ ${#bj[@]} -gt 0 ] && awk -F, "$BJGRID"'
    ($1=="ISO" || $1=="HOT") && ongrid($8) { k = $1","$2","$8; if (!(k in seen)) { seen[k]=1; if ($1 == "ISO") n1[$2]++; else n2[$2]++ } }
    $1=="DONE" { st[$2] = "done"; rt[$2]=$5 }
    $1=="FAIL" { if (!($2 in st)) st[$2]="failed" }
    END { for (d in st) { a = n1[d] + 0; b = n2[d] + 0
            s = (st[d] == "failed") ? "failed" : (a == 11 && b == 11) ? "ok" : "partial"
            print d","s","a","b","rt[d] } }' "${bj[@]}" | sort
} > "$OUT/bjt_thermal/bjt_thermal_samples.csv"
# Output characteristics I_C(V_C) at fixed base current (base driven by current, V_BE computed). Each curve is
# computed in two sweeps from the sample's own V_C (down to 0 V and up to 5 V), so that point appears twice.
bo=("$LOGS"/71_bjto*.log)
{
  echo "sample_id,T0_K,R_K_W,VC0_V,NB_cm3,Nepi_cm3,sweep,IB_A,VC_V,IE_A,IC_A,VBE_V,Tmax_K"
  # a sweep that failed at its first step still leaves one point (I_E = I_C = 0 or NaN): skip it
  [ ${#bo[@]} -gt 0 ] && awk -F, '($1=="OISO" || $1=="OHOT") && $10 != "NaN" && $10 + 0 != 0 {
      sw = ($1 == "OISO") ? "isothermal" : "selfheating"; k = $2","sw","$8","$9
      if (!(k in s)) { s[k]=1; print $2","$3","$4","$5","$6","$7","sw","$8","$9","$10","$11","$13","$14 } }' "${bo[@]}" |
    sort -t, -k1,1 -k7,7 -k8,8g -k9,9g
} > "$OUT/bjt_thermal/bjt_thermal_output.csv"
{
  echo "sample_id,status,n_isothermal,n_selfheating,runtime_s"
  [ ${#bo[@]} -gt 0 ] && awk -F, '
    ($1=="OISO" || $1=="OHOT") && $10 != "NaN" && $10 + 0 != 0 { k = $1","$2","$8","$9; if (!(k in seen)) { seen[k]=1; if ($1 == "OISO") n1[$2]++; else n2[$2]++ } }
    $1=="DONE" { st[$2] = "done"; rt[$2]=$5 }
    $1=="FAIL" { if (!($2 in st)) st[$2]="failed" }
    END { for (d in st) { a = n1[d] + 0; b = n2[d] + 0
            s = (st[d] == "failed") ? "failed" : (a == b && a >= 27) ? "ok" : "partial"
            print d","s","a","b","rt[d] } }' "${bo[@]}" | sort
} > "$OUT/bjt_thermal/bjt_thermal_output_samples.csv"

# ---------- pn diode (1D, cross-section area 4 um^2 = 4e-8 cm^2) ----------
dl=("$LOGS"/20_diode_*.log)
{
  echo "device_id,Na_cm3,Nd_cm3,tau_s,T_K,V_V,I_A,J_A_cm2"
  [ ${#dl[@]} -gt 0 ] && awk -F, '$1=="IV" { k=$2","$7; if (!(k in s)) { s[k]=1; printf "%s,%s,%s,%s,%s,%s,%s,%.6e\n", $2,$3,$4,$5,$6,$7,$8,$8/4e-8 } }' "${dl[@]}" | sort -t, -k1,1 -k6,6g
} > "$OUT/pn_diode/pn_diode_iv.csv"

# ---------- solar cell (1D, area 1 cm^2, so I in mA equals J in mA/cm^2) ----------
sl=("$LOGS"/40_solar_*.log)
{
  echo "device_id,Nbase_cm3,Nemit_cm3,xj_um,W_um,tau_s,suns,T_K,V_V,J_mA_cm2"
  [ ${#sl[@]} -gt 0 ] && awk -F, '$1=="JV" { k=$2","$10; if (!(k in s)) { s[k]=1; printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%.6e\n", $2,$3,$4,$5,$6,$7,$8,$9,$10,$11*1000 } }' "${sl[@]}" | sort -t, -k1,1 -k9,9g
} > "$OUT/solar_cell/solar_cell_jv.csv"
# Figures of merit per cell: Jsc = J(V=0); Voc by linear interpolation of the J=0 crossing;
# Pmax = max(V*J) over the sampled points; FF = Pmax/(Voc*Jsc); eta = Pmax/(100 mW/cm^2 * suns).
awk -F, 'NR==1 { print "device_id,Nbase_cm3,Nemit_cm3,xj_um,W_um,tau_s,suns,T_K,Jsc_mA_cm2,Voc_V,Pmax_mW_cm2,FF,eta"; next }
  {
    if ($1 != cur) { flush(); cur = $1; inp = $2","$3","$4","$5","$6","$7","$8; suns = $7; jsc = ""; voc = ""; pmax = 0; pv = ""; pj = "" }
    v = $9 + 0; j = $10 + 0
    if (v == 0) jsc = j
    if (v * j > pmax) pmax = v * j
    if (pv != "" && voc == "" && pj > 0 && j <= 0) voc = pv + (v - pv) * pj / (pj - j)
    pv = v; pj = j
  }
  function flush() {
    if (cur == "" || jsc == "" || voc == "") return
    printf "%s,%s,%.5g,%.5g,%.5g,%.5g,%.5g\n", cur, inp, jsc, voc, pmax, pmax / (voc * jsc), pmax / (100 * suns)
  }
  END { flush() }' "$OUT/solar_cell/solar_cell_jv.csv" > "$OUT/solar_cell/solar_cell_summary.csv"

for f in "$OUT"/*/*.csv; do printf '%-60s %8d rows\n' "${f#$OUT/}" $(( $(wc -l < "$f") - 1 )); done
