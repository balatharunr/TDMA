# TDMA Schedule Optimizer

A Python tool that turns a set of radio positions into a conflict-free TDMA
(Time Division Multiple Access) schedule: every radio gets one or more time
slots such that no two radios within radio range **or one hop away from each
other** ever transmit at the same time — including the "hidden terminal"
case, where two radios can't hear each other directly but both reach a third
radio that would see a collision.

Built for the Vaan Megam Networks Wireless Developer internship task.
**Part 1** (this repository's `tdma_brain` package) is complete: an exact,
independently-verified, benchmarked scheduler. **Part 2** (a physical EMANE
emulation of the schedule) is a separate, later phase — see
[docs/DESIGN.md](docs/DESIGN.md) for the full write-up of both parts.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# the official 16-node, 300 m grid from the brief
python tdma_schedule.py --file examples/grid16_300m.json
```

Expected output (also reproduced verbatim in [docs/DESIGN.md](docs/DESIGN.md#final-values-the-official-input)):

```
================================================================
               TDMA TOPOLOGY OPTIMIZATION REPORT
================================================================
Total Nodes Processed   : 16
Configured Radio Range  : 500.0 meters
Optimized Frame Length  : 9 unique timeslots (Lower is better)
Scheduling Algorithm    : DSATUR on 2-hop conflict graph (G^2)
Spatial Reuse Factor    : 1.78 nodes/slot (plain TDMA = 1.00)
Lower Bound (proof)     : 9 timeslots (max 2-hop clique = 9, maxdeg+1 = 9)
Optimality              : PROVEN OPTIMAL (frame length = lower bound)
...
```

Exit codes: `0` = schedule verified conflict-free, `1` = verification
failed, `2` = bad input or usage.

## Other useful commands

```powershell
# your own topology, straight on the command line
python tdma_schedule.py '{"Node_01":[0.0,0.0],"Node_02":[300.0,0.0],"Node_03":[600.0,0.0]}'

# the proof, step by step: every heuristic's slot count, the lower bound, the exact search
python tdma_schedule.py --file examples/ring16_r1150m.json --verbose

# why distance-2 colouring is required: colour the radio graph alone and watch it fail verification
python tdma_schedule.py --file examples/grid16_300m.json --algorithm naive-d1

# a topology picture (coloured by slot) and a slot x radio frame-timeline picture,
# plus the schedule as JSON and the Boolean matrix as CSV
python tdma_schedule.py --file examples/grid16_300m.json ^
  --plot results/topology.png --timeline results/timeline.png --highlight Node_06 ^
  --export-json results/schedule.json --export-csv results/matrix.csv

# heuristic comparison across a grid of random topologies (n x avg_degree x seed)
python bench/benchmark.py --quick     # 20 instances, a few seconds
python bench/benchmark.py             # 160 instances, ~15 seconds on this machine
```

Run `python tdma_schedule.py --help` for the full flag reference.

## Project layout

```
tdma_brain/          the library: input -> radio graph -> conflict graph -> schedule -> proof
  topology.py           radio graph G, conflict graph H = G^2, hidden-terminal detection
  heuristics.py          DSATUR, RLF, smallest-last, largest-first (Welsh-Powell), greedy, iterated greedy
  bounds.py               lower bound: max(maxdeg(G)+1, max clique of H)
  exact.py                 DSATUR-ordered branch-and-bound exact search, symmetry-broken, time-limited
  optimizer.py              orchestrates all of the above into one verified, proven result
  schedule.py                the Schedule model + the schedule.json (v1) and CSV exports
  verify.py                    independent BFS-based conflict checker (never touches H)
  viz.py                        topology and frame-timeline plots (matplotlib)
  report.py                     the ASCII report and --verbose output
  cli.py                        argparse, orchestration, exit codes
bench/benchmark.py    the evidence-pack generator: runs the grid, writes results/
examples/             9 golden topologies with independently pre-computed correct answers
tests/                657 tests: golden-value, property-based and CLI-level
results/              benchmark.csv/.md, plots, and the official grid's schedule.json/CSV — all generated, not hand-written
docs/DESIGN.md        the full write-up: problem, design decisions, algorithms, verification,
                       final values, benchmark results, Part 2 (added later)
```

## Results summary

Full numbers, all benchmark tables and the design rationale are in
[docs/DESIGN.md](docs/DESIGN.md). Headline results from the full
160-instance benchmark (`n` in {16, 32, 64, 128} x `avg_degree` in {4, 8} x
20 seeds each, see `results/benchmark_summary.md`):

- **100%** of the 160 random instances, and all 9 hand-built golden
  examples, are proven optimal — the schedule's frame length equals the
  independently-computed lower bound.
- **Zero verification failures** anywhere: every schedule the optimizer
  produces passes the independent BFS-based verifier.
- Spatial reuse (`n / K`) grows with network size at fixed density, from
  ~2.5x at n=16 up to ~13x at n=128 (avg_degree=4) — exactly as expected
  from the theory of spatial TDMA reuse.
- A naive distance-1-only colouring (ignoring hidden terminals) is
  conflict-free in **0%** of instances — the concrete evidence for why
  distance-2 colouring is required, not optional.

## Testing

```powershell
pytest -q
```

657 tests: 9 hand-verified golden topologies (values computed
independently, by exact search, before any code was written) plus
property-based and CLI-level tests for every module. See
[docs/DESIGN.md](docs/DESIGN.md#verification--testing-strategy) for the
full testing and verification strategy.
