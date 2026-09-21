# What is running while Claude is off — 2026-09-21 to 2026-09-23

Nothing here needs Claude. It is shell and python, driven by a supervisor loop.

## Check it in one line

    tail -20 /Volumes/seagate/local_noise_run/supervisor/supervisor.log

A line every 15 minutes reading `alive, stores: N usable...` means it is
healthy. If that line stops arriving, the supervisor is dead rather than busy.

    cat /Volumes/seagate/local_noise_run/supervisor/alerts.log

Only things worth acting on: a job that FAILED, one OVERDUE past four hours, a
cache store moved aside or unreadable. Nothing truncates this file.

## What is running

27 jobs in `supervisor/jobs.tsv`, three at a time, covering all ten methods:
n02 x3, n07 x3, n09, n11, n12 x3, n17 x3, n18 x3, n20, n21 x3, n23 x4.

Each job exits at once if its result file already exists, so a finished job
re-runs after its cool-off and costs seconds. Work is never repeated.

## The three guards, and why each exists

  CONCURRENCY   at most 3 compute jobs. A load ceiling was tried first and was
                the wrong rule -- load ran 200 to 840 all night from another
                user's work, so every job sat held and the queue could only
                drain if that stopped. A run that went ahead at load 486
                finished in 9,626 seconds. A tenth of a core is slow, not zero.

  MEMORY        no NEW job starts while system-wide free memory is under 20%.
                Read from `memory_pressure`, NOT from vm_stat's free pages --
                that reported 155MB on a machine with 63% available, because
                macOS keeps all spare RAM as cache, and using it would have
                held everything for ever. Running jobs are never killed for
                this: part-done work is wasted twice.

  DISK          no new job starts with under 20GB free on /Volumes/seagate.

## Everything large is on the external drive

`KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache`, exported by the supervisor for
every job it starts. This matters: `rep_cache.py` falls back to `~/.cache/kirby`
whenever that variable is unset, and the boot disk has 38GB free against 1.5TB
on the external one. A job that forgot to export it would quietly fill the boot
disk with representation matrices.

## If it stops

A watchdog (`supervisor/watchdog.sh`) checks every 60 seconds and restarts the
supervisor if it died. Tested: killed the supervisor, it was back in 75 seconds.

⛔ NEITHER SURVIVES A REBOOT. After one, run:

    cd /Volumes/seagate/local_noise_run/supervisor && nohup ./watchdog.sh >/dev/null 2>&1 &

⛔ WHY NOT launchd, which would survive a reboot. It cannot write to
/Volumes/seagate: macOS refuses with "Operation not permitted", because an
external volume needs a Full Disk Access grant that a background agent does not
inherit from Terminal. Tried twice on 2026-09-21 -- EX_CONFIG first on the
working directory and log paths, then this. The plist is parked at
`~/Library/LaunchAgents/com.kirby.noise-supervisor.plist.needs-full-disk-access`.
To make it work: System Settings, Privacy & Security, Full Disk Access, add
/bin/bash. Then rename the plist back and `launchctl load` it. Your call --
granting bash full disk access is not a small thing.

## What to read first when Claude is back

  tests/experiments/noise/OPEN_THREADS.md   every loose end, section F first
  python tests/experiments/noise/coverage.py   what has been optimised
  supervisor/alerts.log                     anything that failed meanwhile

Section F is the one that matters: a random drop of the same size wins as many
folds as the best cleaning arm, so "folds won against no cleaning" cannot carry
a cleaning claim on its own. That applies to n02, n07, n11, n17, n18, n20 and
n23 -- every cleaning method here is scored that way.
