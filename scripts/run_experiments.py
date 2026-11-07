"""Run every experiment for the care-facility scheduling study and save CSVs + plots.

    python scripts/run_experiments.py         # everything (noise study takes ~10-15 min)
    python scripts/run_experiments.py --skip-noise  # everything except the noise study (~2 min)
    python scripts/run_experiments.py --only-noise  # just the noise study

Sizes: 4, 6 and 8 data qubits (2, 3, 4 staff). The 10-qubit case is skipped on purpose.
Simulator wall-clock time is NEVER used as evidence of quantum speed - only query counts,
gate counts and success probabilities are.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse, csv, json, platform, random
from math import asin, sqrt, sin
from pathlib import Path
from time import perf_counter

import numpy as np
import qiskit, qiskit_aer, ortools
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from scheduling import Problem, build_oracle, grover_circuit, known_iterations
from instances import INSTANCES, care_facility, describe
from classical import brute_force_random, backtracking, cpsat, real_size_cpsat

OUT = ROOT / 'results' / 'experiments'
SEED = 2026
SHOTS = 2000


def save(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
    print(f'  saved results/experiments/{name}')


def count_solutions(problem):
    return [x for x in range(2 ** problem.n) if problem.feasible(x)]


def compile_grover(problem, oracle, k, measure):
    # optimization_level=3: give the quantum side the best standard compiler (fair to quantum).
    return transpile(grover_circuit(problem, oracle, k, measure=measure),
                     basis_gates=['u', 'cx'], optimization_level=3, seed_transpiler=0)


def exact_success(problem, oracle, k):
    """Exact probability that measuring gives a valid schedule (no sampling noise)."""
    qc = compile_grover(problem, oracle, k, measure=False)
    qc.save_statevector()
    probs = np.abs(np.asarray(AerSimulator(method='statevector').run(qc).result().get_statevector())) ** 2
    n = problem.n
    # Index = data bits (low n) + work bits (high). Count only clean work bits.
    p_valid = sum(probs[x] for x in range(2 ** n) if problem.feasible(x))
    p_dirty = probs[2 ** n:].sum()
    return float(p_valid), float(p_dirty), qc


def theory_success(n, M, k):
    return sin((2 * k + 1) * asin(sqrt(M / 2 ** n))) ** 2


# ---------------------------------------------------------------------------------------
def experiment_instances():
    print('\n[1] Instances and their valid schedules')
    rows, sol_rows = [], []
    for n, make in INSTANCES.items():
        p = make(); sols = count_solutions(p)
        rows.append(dict(data_qubits=n, staff=n // 2, rules=len(p.rules), N=2 ** n, M=len(sols),
                         valid_fraction=len(sols) / 2 ** n, rule_names='; '.join(r.name for r in p.rules)))
        for x in sols:
            sol_rows.append(dict(data_qubits=n, bitstring_int=x, schedule=describe(p, x)))
        print(f'  {n} qubits: {len(p.rules)} rules, M={len(sols)} of N={2**n} ({len(sols)/2**n:.1%})')
    save('instances.csv', rows); save('valid_schedules.csv', sol_rows)


def experiment_iteration_sweep():
    print('\n[2] Iteration sweep: exact success probability vs. theory')
    rows = []
    for n, make in INSTANCES.items():
        p = make(); M = len(count_solutions(p)); o = build_oracle(p)
        k_best = known_iterations(n, M)
        for k in range(0, 3 * k_best + 2):
            pv, dirty, _ = exact_success(p, o, k)
            rows.append(dict(data_qubits=n, M=M, N=2 ** n, iterations=k, optimal_iterations=k_best,
                             exact_success=pv, theory_success=theory_success(n, M, k),
                             dirty_work_qubit_probability=dirty))
            assert abs(pv - theory_success(n, M, k)) < 1e-6 and dirty < 1e-9
        print(f'  {n} qubits: optimal k={k_best}, P={theory_success(n, M, k_best):.3f} (simulation matches theory)')
    save('iteration_sweep.csv', rows)


def experiment_shots_and_cost():
    print(f'\n[3] Measured runs ({SHOTS} shots) and circuit cost')
    shots_rows, cost_rows = [], []
    sim = AerSimulator(method='statevector')
    for n, make in INSTANCES.items():
        p = make(); sols = count_solutions(p); M = len(sols); o = build_oracle(p)
        k = known_iterations(n, M)
        qc = compile_grover(p, o, k, measure=True)
        counts = sim.run(qc, shots=SHOTS, seed_simulator=SEED).result().get_counts()
        good = 0
        for key, c in counts.items():
            x = int(key.replace(' ', ''), 2)
            data, work = x & (2 ** n - 1), x >> n
            assert work == 0, 'oracle left a work qubit dirty'
            if p.feasible(data):
                good += c
            shots_rows.append(dict(data_qubits=n, iterations=k, outcome_int=data, valid=p.feasible(data),
                                   schedule=describe(p, data), count=c))
        one = compile_grover(p, o, 1, measure=False)
        P = good / SHOTS
        cost_rows.append(dict(
            data_qubits=n, rules=len(p.rules), total_qubits=qc.num_qubits, work_qubits=o.ancillas,
            N=2 ** n, M=M, iterations=k,
            cx_per_iteration=one.count_ops().get('cx', 0), cx_total=qc.count_ops().get('cx', 0),
            single_qubit_total=qc.count_ops().get('u', 0), depth=qc.depth(),
            measured_success=P, random_guess_success=M / 2 ** n,
            # Repeat until success: expected oracle calls and expected checks of the output.
            expected_oracle_calls=k / P, expected_total_queries=(k + 1) / P))
        print(f'  {n} qubits: k={k}, measured success {P:.1%} vs random guess {M/2**n:.1%}; '
              f'{cost_rows[-1]["cx_total"]} CX gates, {qc.num_qubits} qubits')
    save('measured_outcomes.csv', shots_rows); save('circuit_cost.csv', cost_rows)
    return cost_rows


def experiment_classical(cost_rows):
    print('\n[4] Classical baselines on the same instances')
    rng = random.Random(SEED)
    rows = []
    q = {r['data_qubits']: r for r in cost_rows}
    for n, make in INSTANCES.items():
        p = make(); M = len(count_solutions(p)); N = 2 ** n
        bf = [brute_force_random(p, rng) for _ in range(5000)]
        bt = [backtracking(p, rng)[1] for _ in range(2000)]
        cs = cpsat(p)
        rows += [
            dict(data_qubits=n, N=N, M=M, method='brute force (random order)', unit='full schedules checked',
                 mean=np.mean(bf), std=np.std(bf), formula=(N + 1) / (M + 1), trials=len(bf)),
            dict(data_qubits=n, N=N, M=M, method='Grover (repeat until valid)', unit='oracle calls + output checks',
                 mean=q[n]['expected_total_queries'], std='', formula='(k+1)/P', trials=SHOTS),
            dict(data_qubits=n, N=N, M=M, method='backtracking', unit='partial schedules visited',
                 mean=np.mean(bt), std=np.std(bt), formula='', trials=len(bt)),
            dict(data_qubits=n, N=N, M=M, method='CP-SAT', unit='search branches',
                 mean=cs['branches'], std='', formula='', trials=1, wall_seconds=cs['wall_seconds']),
        ]
        print(f'  {n} qubits: brute {np.mean(bf):.1f} | Grover {q[n]["expected_total_queries"]:.1f} | '
              f'backtracking {np.mean(bt):.1f} | CP-SAT {cs["branches"]} branches in {cs["wall_seconds"]*1000:.1f} ms')
    save('classical_comparison.csv', rows)


def experiment_constraint_ladder():
    print('\n[5] Constraint ladder (8 qubits): add rules one at a time')
    full = care_facility(4); rows = []
    for m in range(1, len(full.rules) + 1):
        p = Problem(full.assignments, full.rules[:m], f'ladder_{m}')
        M = len(count_solutions(p)); o = build_oracle(p)
        k = known_iterations(p.n, M)
        one = compile_grover(p, o, 1, measure=False)
        rows.append(dict(rules=m, newest_rule=full.rules[m - 1].name, M=M, valid_fraction=M / 256,
                         iterations=k, total_qubits=one.num_qubits,
                         cx_per_iteration=one.count_ops().get('cx', 0),
                         cx_total=one.count_ops().get('cx', 0) * k,
                         success=theory_success(p.n, M, k) if 0 < M < 256 else (1.0 if M == 256 else 0.0)))
        print(f'  {m:2d} rules (+{full.rules[m-1].name}): M={M}, k={k}, CX/iteration={rows[-1]["cx_per_iteration"]}')
    save('constraint_ladder.csv', rows)


def experiment_real_size():
    print('\n[6] CP-SAT at realistic size (quantum would need these many qubits for data alone)')
    rows = []
    for staff in (30, 60, 120, 250):
        r = real_size_cpsat(staff, optimize=False); rows.append(r)
        print(f'  {staff} staff x 7 days x 3 shifts = {r["binary_variables"]} yes/no decisions '
              f'(2^{r["binary_variables"]} schedules): feasible schedule in {r["wall_seconds"]} s')
    save('real_size_cpsat.csv', rows)


def experiment_noise(rates=(1e-2, 1e-3, 1e-4, 1e-5)):
    print('\n[7] Noise study: depolarizing error on every gate')
    shots_for = {4: 2000, 6: 1000, 8: 400}
    rows = []
    for n, make in INSTANCES.items():
        p = make(); M = len(count_solutions(p)); o = build_oracle(p)
        k = known_iterations(n, M)
        qc = compile_grover(p, o, k, measure=True)
        cx, u = qc.count_ops().get('cx', 0), qc.count_ops().get('u', 0)
        ideal = theory_success(n, M, k)
        for rate in (0.0,) + tuple(rates):
            nm = NoiseModel()
            if rate:
                nm.add_all_qubit_quantum_error(depolarizing_error(rate / 10, 1), ['u'])
                nm.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ['cx'])
            sim = AerSimulator(method='statevector', noise_model=nm if rate else None)
            shots = shots_for[n]
            t = perf_counter()
            counts = sim.run(qc, shots=shots, seed_simulator=SEED).result().get_counts()
            good = sum(c for key, c in counts.items()
                       if p.feasible(int(key.replace(' ', ''), 2) & (2 ** n - 1)))
            P = good / shots
            # Simple model: circuit survives error-free with prob F, otherwise output ~ random.
            F = (1 - rate) ** cx * (1 - rate / 10) ** u
            rows.append(dict(data_qubits=n, two_qubit_error_rate=rate, shots=shots, iterations=k,
                             cx_total=cx, measured_success=P,
                             std_error=sqrt(max(P * (1 - P), 1e-12) / shots),
                             ideal_success=ideal, random_guess_success=M / 2 ** n,
                             model_no_error_probability=F,
                             model_success=F * ideal + (1 - F) * M / 2 ** n,
                             sim_seconds=round(perf_counter() - t, 1)))
            print(f'  {n} qubits, error {rate:g}: success {P:.1%} (ideal {ideal:.1%}, random {M/2**n:.1%}, '
                  f'model {rows[-1]["model_success"]:.1%})', flush=True)
    save('noise.csv', rows)


def environment(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'environment{"_noise" if args.only_noise else ""}.json').write_text(json.dumps(dict(
        python=platform.python_version(), qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
        ortools=ortools.__version__, numpy=np.__version__, platform=platform.platform(),
        simulator='Qiskit Aer AerSimulator, statevector method, CPU',
        transpile='basis u+cx, optimization_level=3, seed_transpiler=0',
        noise='depolarizing: rate p on every CX, p/10 on every single-qubit u gate',
        seed=SEED, shots=SHOTS, arguments=vars(args)), indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-noise', action='store_true')
    ap.add_argument('--only-noise', action='store_true')
    args = ap.parse_args()
    environment(args)
    if not args.only_noise:
        experiment_instances()
        experiment_iteration_sweep()
        cost = experiment_shots_and_cost()
        experiment_classical(cost)
        experiment_constraint_ladder()
        experiment_real_size()
    if not args.skip_noise:
        experiment_noise()
    print('\nDone. Make plots with: python scripts/make_plots.py')
