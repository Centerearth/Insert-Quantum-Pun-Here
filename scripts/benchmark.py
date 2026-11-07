"""Original synthetic benchmark: python scripts/benchmark.py --trials 20 --seed 2026"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse
import csv
import json
from math import ceil, sqrt
from pathlib import Path
import platform
from statistics import mean
import numpy as np
import qiskit
import qiskit_aer
from scheduling import (GroverEngine, classical_search, exhaustive_audit, fixture,
                        integer, known_iterations, load_problem)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', nargs='+', type=int, default=[3, 5, 10])
    parser.add_argument('--constraints', nargs='+', type=int, default=[5, 10, 25])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--family', choices=['quarter', 'unique', 'none', 'all'], default='quarter')
    parser.add_argument('--unknown-m', action='store_true',
                        help='Do not give audit M to Grover; use bounded random lengths')
    parser.add_argument('--problem', type=Path, help='Use this JSON instead of synthetic grid')
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'benchmark')
    args = parser.parse_args()
    try:
        integer(args.trials, 'trials', 1, 100000)
        integer(args.seed, 'seed', 0, 2**32-1)
        integer(args.attempts, 'attempts', 1, 1000)
        for n in args.qubits:
            integer(n, 'data qubits', 1, 10)
        for m in args.constraints:
            integer(m, 'active constraints', 0, 25)
        if len(set(args.qubits)) != len(args.qubits) or len(set(args.constraints)) != len(args.constraints):
            raise ValueError('benchmark axes must not contain duplicate values')
        problems = ([(load_problem(args.problem), None)] if args.problem else
                    [fixture(n, m, args.family) for n in args.qubits for m in args.constraints])
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, setups, summary = [], [], []
    for case, (problem, expected) in enumerate(problems):
        audit = exhaustive_audit(problem)  # At most 1024 candidates; NEVER hidden.
        if expected is not None and audit['marked'] != expected:
            raise AssertionError('synthetic fixture count disagrees with exhaustive checker')
        engine = GroverEngine(problem)
        metadata = dict(problem=problem.name, data_qubits=problem.n,
                        active_constraints=len(problem.rules), N=2**problem.n,
                        M=audit['marked'])
        setup = dict(**metadata, **audit, expected_M=expected,
                     oracle_build_seconds=engine.oracle.compile_seconds,
                     synthesis_predicate_evaluations=engine.oracle.local_table_evaluations,
                     synthesis_minterms=engine.oracle.minterms,
                     oracle_ancillas=engine.oracle.ancillas,
                     oracle_qubits=engine.oracle.circuit.num_qubits,
                     oracle_abstract_size=engine.oracle.circuit.size(),
                     oracle_abstract_depth=engine.oracle.circuit.depth())
        setups.append(setup)
        case_rows = []
        for trial in range(args.trials):
            # Independent deterministic streams for quantum lengths/shots and random baseline.
            qseed, cseed, eseed = np.random.SeedSequence([args.seed, case, trial]).spawn(3)
            qrng = np.random.default_rng(qseed)
            marked = None if args.unknown_m else audit['marked']
            plan = ([int(qrng.integers(0, ceil(sqrt(2**problem.n)))) for _ in range(args.attempts)]
                    if marked is None else [known_iterations(problem.n, marked)]*args.attempts)
            # Fix the common cap BEFORE seeing results, to avoid outcome-dependent budgets.
            budget = sum(plan)+args.attempts
            qrow = engine.run(qrng, marked=marked, attempts=args.attempts, iteration_plan=plan)
            qrow['decision_query_cap'] = budget
            crow = classical_search(problem, np.random.default_rng(cseed), budget=budget)
            crow['decision_query_cap'] = budget
            erow = classical_search(problem, np.random.default_rng(eseed), exhaustive=True)
            for row in (qrow, crow, erow):
                row.update(metadata, trial=trial)
                case_rows.append(row)
        rows.extend(case_rows)
        for method in ('grover', 'random_without_replacement', 'exhaustive_first'):
            subset = [r for r in case_rows if r['method'] == method]
            summary.append(dict(**metadata, method=method, trials=args.trials,
                                successes=sum(r['success'] for r in subset),
                                success_rate=mean(r['success'] for r in subset),
                                mean_oracle_calls=mean(r['logical_oracle_calls'] for r in subset),
                                mean_candidate_evaluations=mean(r['candidate_evaluations'] for r in subset),
                                mean_constraint_evaluations=mean(r['constraint_evaluations'] for r in subset),
                                mean_reversible_rule_blocks=mean(r['reversible_rule_blocks'] for r in subset),
                                mean_search_wall_seconds=mean(r['search_wall_seconds'] for r in subset),
                                mean_simulator_seconds=mean(r['simulator_seconds'] for r in subset),
                                total_transpile_seconds=sum(r['transpile_seconds'] for r in subset),
                                ancillas=engine.oracle.ancillas if method == 'grover' else 0))
        print(f"{problem.name}: M={audit['marked']}/{2**problem.n}; "
              f"Grover successes={sum(r['success'] for r in case_rows if r['method']=='grover')}/{args.trials}", flush=True)
    write_csv(args.out / 'trials.csv', rows)
    write_csv(args.out / 'setup.csv', setups)
    write_csv(args.out / 'summary.csv', summary)
    metadata = dict(python=platform.python_version(), qiskit=qiskit.__version__,
                    qiskit_aer=qiskit_aer.__version__, numpy=np.__version__,
                    simulator='AerSimulator(statevector), CPU, 1 thread, noiseless',
                    arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    timing='Oracle build and exhaustive audit are one-time setup; transpilation cache shared across trials; each attempt uses a fresh one-shot Aer job.')
    (args.out / 'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('\n n   m method                       success   oracle   candidates   sim seconds')
    for row in summary:
        print(f"{row['data_qubits']:2d} {row['active_constraints']:3d} {row['method']:28s} "
              f"{row['success_rate']:7.1%} {row['mean_oracle_calls']:8.2f} "
              f"{row['mean_candidate_evaluations']:12.2f} {row['mean_simulator_seconds']:12.6f}")
    print(f'\nCSV files written to {args.out.resolve()}')


if __name__ == '__main__':
    main()
