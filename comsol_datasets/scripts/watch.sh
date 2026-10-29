#!/usr/bin/env bash
# Waits until a job fails, the runner exits, or MAXWAIT seconds pass; then prints a status summary.
RUN="${RUN:-C:/Users/janze/comsol_runs}"
MAXWAIT="${1:-2700}"
nfail0=$(ls "$RUN/failed" | wc -l)
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt "$MAXWAIT" ]; do
  [ "$(ls "$RUN/failed" | wc -l)" -gt "$nfail0" ] && { echo "REASON: new failed job"; break; }
  grep -q "runner exiting" "$RUN/runner.log" && { echo "REASON: runner exited"; break; }
  sleep 30
done
echo "time: $(date '+%F %T')"
for g in 10_mos 11_mos 20_diode 30_mos 35_mos 40_solar; do
  printf '%-10s queued %3d  running %2d  done %3d  failed %2d  skipped %3d\n' "$g" \
    "$(ls "$RUN/queue" | grep -c "^$g")" "$(ls "$RUN/running" | grep -c "^$g")" \
    "$(ls "$RUN/done" | grep -c "^$g")" "$(ls "$RUN/failed" | grep -c "^$g")" "$(ls "$RUN/skipped" | grep -c "^$g")"
done
echo "device FAIL lines: $(cat "$RUN"/logs/*.log 2>/dev/null | grep -cE '^(FAIL|FAILVG|FAILREV),')"
tail -4 "$RUN/runner.log"
