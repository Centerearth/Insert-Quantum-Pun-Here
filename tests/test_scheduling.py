"""Tests check mathematical semantics, phases, clean work qubits and boundaries."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('qiskit')
pytest.importorskip('qiskit_aer')
from qiskit import transpile
from qiskit_aer import AerSimulator
from scheduling import (Assignment, Counts, GroverEngine, Problem, Rule,
                        build_oracle, classical_search, exhaustive_audit,
                        fixture, integer, known_iterations, load_problem)


def state_after(problem, basis=None):
    oracle = build_oracle(problem)
    qc = oracle.circuit.copy_empty_like()
    qc.global_phase = 0
    if basis is None:
        qc.h(range(problem.n))
    else:
        for i in range(problem.n):
            if (basis >> i) & 1:
                qc.x(i)
    qc.compose(oracle.circuit, inplace=True)
    qc.save_statevector()
    backend = AerSimulator(method='statevector', max_parallel_threads=1)
    compiled = transpile(qc, backend, optimization_level=0, seed_transpiler=0)
    return np.asarray(backend.run(compiled).result().get_statevector())


@pytest.mark.parametrize('family', ['quarter', 'unique', 'none', 'all'])
@pytest.mark.parametrize('m', [1, 5, 10, 25])
def test_oracle_truth_table_and_clean_work(family, m):
    problem, expected = fixture(3, m, family)
    assert exhaustive_audit(problem)['marked'] == expected
    vector = state_after(problem)
    target = np.zeros_like(vector)
    for x in range(8):
        target[x] = (-1 if problem.feasible(x) else 1) / np.sqrt(8)
    np.testing.assert_allclose(vector, target, atol=1e-9)


def test_all_basis_inputs_with_mixed_supported_rules():
    assignments = [Assignment('E', f'S{i}', 'L', 'R', 1) for i in range(3)]
    rules = [Rule('count', 'count', (0, 1, 2), {'lower': 1, 'upper': 2}),
             Rule('weighted', 'weighted', (0, 2), {'weights': [2, 5], 'lower': 2, 'upper': 5}),
             Rule('paired', 'equal', (0, 1), {}),
             Rule('forbidden', 'pattern', (1, 2), {'pattern': [1, 1]})]
    problem = Problem(assignments, rules)
    for x in range(8):
        target = np.zeros(2**build_oracle(problem).circuit.num_qubits, dtype=complex)
        target[x] = -1 if problem.feasible(x) else 1
        np.testing.assert_allclose(state_after(problem, x), target, atol=1e-9)


def test_ten_qubits_and_25_rules_phase():
    problem, expected = fixture(10, 25, 'unique')
    assert expected == 1
    vector = state_after(problem)
    target = np.zeros_like(vector)
    target[:1024] = 1/32
    target[0] = -1/32
    np.testing.assert_allclose(vector, target, atol=1e-9)
    assert build_oracle(problem).ancillas == 6


def test_example_and_classical_accounting():
    problem = load_problem(Path(__file__).with_name('example.json'))
    valid = [x for x in range(16) if problem.feasible(x)]
    assert valid == [6]  # Ben S0 and Ada S1.
    assert [(a.employee, a.shift) for a in problem.decode(6)] == [('Ben', 'S0'), ('Ada', 'S1')]
    counts = Counts()
    assert not problem.feasible(1, counts)
    assert (counts.candidates, counts.constraints) == (1, 1)
    counts = Counts()
    problem.feasible(1, counts, short_circuit=False)
    assert counts.constraints == 5
    assert classical_search(problem, np.random.default_rng(4), exhaustive=True)['candidate_evaluations'] == 7


def test_no_rules_and_unknown_failure_status():
    p, _ = fixture(2, 1)
    empty = Problem(p.assignments, [])
    assert exhaustive_audit(empty)['marked'] == 4
    assert build_oracle(empty).ancillas == 0
    np.testing.assert_allclose(state_after(empty), [-.5]*4, atol=1e-9)
    none, _ = fixture(2, 5, 'none')
    row = GroverEngine(none).run(np.random.default_rng(2), marked=None, attempts=2)
    assert row['status'] == 'no_verified_sample'
    assert row['logical_oracle_calls'] == row['grover_iterations']
    assert row['candidate_evaluations'] == 2
    assert row['reversible_rule_blocks'] == 20*row['logical_oracle_calls']
    classical = classical_search(none, np.random.default_rng(3), exhaustive=True)
    assert classical['status'] == 'proven_unsat'
    assert classical['candidate_evaluations'] == 4


def test_grover_known_dense_case():
    problem, M = fixture(3, 5)
    assert known_iterations(3, M) == 1
    row = GroverEngine(problem).run(np.random.default_rng(5), marked=M)
    assert row['success']
    assert row['logical_oracle_calls'] == 1
    assert row['candidate_evaluations'] == 1


@pytest.mark.parametrize('n,m', [(11, 5), (0, 5), (3, 26), (3, -1)])
def test_limits(n, m):
    with pytest.raises(ValueError):
        fixture(n, m)


@pytest.mark.parametrize('args', [
    ('bad', 'callable', (0,), {}),
    ('bad', 'count', (0,), {'lower': -1, 'upper': 2}),
    ('bad', 'count', (0,), {'lower': 2, 'upper': 1}),
    ('bad', 'equal', (0,), {}),
    ('bad', 'pattern', (0,), {'pattern': [2]}),
    ('bad', 'weighted', (0,), {'lower': 0, 'upper': 1, 'weights': []}),
    ('bad', 'count', (0, 0), {'lower': 0, 'upper': 1}),
])
def test_invalid_rule(args):
    with pytest.raises(ValueError):
        Rule(*args)


def test_reject_out_of_range_and_duplicates():
    p, _ = fixture(3, 5)
    with pytest.raises(ValueError):
        Problem(p.assignments, [Rule('bad', 'equal', (0, 3), {})])
    with pytest.raises(ValueError):
        Problem(p.assignments, [p.rules[0], p.rules[0]])
    with pytest.raises(ValueError):
        integer(True, 'n', 0, 10)
