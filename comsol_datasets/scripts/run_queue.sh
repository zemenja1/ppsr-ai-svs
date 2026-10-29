#!/usr/bin/env bash
# Runs COMSOL batch jobs from a queue directory, keeping up to MAXJOBS running.
#
# Job layout: $RUN/queue/<name>/ contains <Class>.class and a file "limit" (max runtime, s).
# Jobs start in lexical order of <name>; the prefix before the first "_" is the job group.
# $RUN/deadlines.txt holds "<group> <epoch>" lines: jobs of a group are not started after
# that time (they are moved to $RUN/skipped). The file is re-read on every loop, so it can
# be edited while the queue runs. Finished jobs move to done/ (or failed/), logs go to logs/.
# The runner exits when the queue is empty, nothing is running and $RUN/STOP_WHEN_EMPTY exists.

RUN="${RUN:-C:/Users/janze/comsol_runs}"
MAXJOBS="${MAXJOBS:-4}"
NP="${NP:-3}"
COMSOL="/c/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolbatch.exe"
mkdir -p "$RUN/queue" "$RUN/running" "$RUN/done" "$RUN/failed" "$RUN/skipped" "$RUN/logs" "$RUN/tmp"

log() { echo "$(date '+%F %T') $*" >> "$RUN/runner.log"; }

deadline_for() {
  local g="$1"
  [ -f "$RUN/deadlines.txt" ] || { echo 9999999999; return; }
  awk -v g="$g" '$1 == g { print $2; f = 1 } END { if (!f) print 9999999999 }' "$RUN/deadlines.txt"
}

start_job() {
  local name="$1" dir="$RUN/running/$1"
  mv "$RUN/queue/$name" "$dir" || return
  local cls limit
  cls=$(basename "$(ls "$dir"/*.class | head -1)" .class)
  limit=$(cat "$dir/limit" 2>/dev/null || echo 3600)
  log "START $name ($cls, limit ${limit}s)"
  (
    cd "$dir" || exit 1
    timeout --kill-after=60 "$limit" "$COMSOL" -inputfile "$cls.class" -nosave -np "$NP" \
      -tmpdir "$RUN/tmp/$name" > "$RUN/logs/$name.log" 2>&1
    rc=$?
    if grep -q "^WORKER_FINISHED" "$RUN/logs/$name.log"; then dest=done; else dest=failed; fi
    mv "$dir" "$RUN/$dest/$name"
    echo "$(date '+%F %T') END $name rc=$rc -> $dest" >> "$RUN/runner.log"
  ) &
}

log "runner started (MAXJOBS=$MAXJOBS NP=$NP)"
while true; do
  running=$(ls "$RUN/running" | wc -l)
  pending=$(ls "$RUN/queue" | sort | head -1)
  if [ -n "$pending" ] && [ "$running" -lt "$MAXJOBS" ]; then
    group="${pending%%_*}"
    if [ "$(date +%s)" -gt "$(deadline_for "$group")" ]; then
      mv "$RUN/queue/$pending" "$RUN/skipped/$pending"
      log "SKIP $pending (group $group past deadline)"
      continue
    fi
    start_job "$pending"
    sleep 5
    continue
  fi
  if [ -z "$pending" ] && [ "$running" -eq 0 ] && [ -f "$RUN/STOP_WHEN_EMPTY" ]; then
    log "queue empty, runner exiting"
    break
  fi
  sleep 20
done
