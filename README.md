# Grover search for staff scheduling

**Team Insert Quantum Pun Here** · IQC 2026, JustBuild track

Should Master Planner, a scheduling outsourcer, invest in quantum scheduling? We ran Grover's search on a care-facility staffing problem and compared it with classical solvers. **Short answer: pass for now.**

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q              # 42 tests
jupyter notebook grover_scheduling.ipynb   # main deliverable, ~12 min to Run All
```

Or without the notebook: `python scripts/run_experiments.py` (CSVs in `results/experiments/`), then `python scripts/make_plots.py` (charts in `results/plots/`).

Everything runs on a CPU with Qiskit Aer's statevector simulator. No quantum hardware was used.

## Layout

```
grover_scheduling.ipynb   executed notebook (main deliverable)
src/
  scheduling.py           rules -> oracle (build_oracle) -> Grover circuit (grover_circuit)
  instances.py            care-facility samples: 2/3/4 staff x Day/Night = 4/6/8 qubits
  classical.py            brute force, backtracking, CP-SAT, real-size CP-SAT
scripts/
  run_experiments.py      script version of the experiments
  make_plots.py           slide charts
  benchmark.py            original synthetic benchmark (kept for reference)
tests/                    pytest suite + fixtures
results/
  notebook/               graphs and CSVs from the notebook
  experiments/            CSVs from run_experiments.py
  plots/                  charts used in the slides
```

## The problem

One qubit per (person, shift) decision. 4 staff × 2 shifts = 8 qubits = 256 schedules, and 11 rules leave 3 valid ones. Rules are written as data (`pattern`, `count`, `weighted`, `equal`). Each rule becomes a gate that flags a violation, a counter adds up the flags, and the oracle flips the sign only when the count is 0.

## Results (8 qubits)

- Grover finds a valid schedule 99.7% of the time, and the simulation matches theory.
- Queries to find a valid schedule: brute force ~64, backtracking ~14, Grover ~8. CP-SAT solves it in presolve.
- But each Grover iteration costs ~2,300 two-qubit gates (16,537 in total).
- With 1-in-1,000 gate errors (today's best hardware), success drops to ~1%, the same as guessing.
- CP-SAT solves a real 30-staff week (2⁶³⁰ schedules) in milliseconds.

## Caveats

Simulator run time isn't evidence of quantum speed, so we compare query and gate counts instead. The best iteration count uses the known number of valid schedules. The noise model is generic depolarizing noise, not a specific device.
