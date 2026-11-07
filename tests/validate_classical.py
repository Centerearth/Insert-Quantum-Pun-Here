"""Standard-library test runner; needs NumPy but not Qiskit or pytest.

This checks classical semantics and fixtures. It does NOT validate Qiskit/Aer.
Run test_scheduling.py with pytest for quantum circuit validation.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import csv
import json
from pathlib import Path
import unittest
import numpy as np
from scheduling import (Assignment, Counts, Problem, Rule, classical_search,
                        exhaustive_audit, fixture, integer, known_iterations,
                        load_problem)


class ClassicalTests(unittest.TestCase):
    def test_all_controlled_counts(self):
        for family in ('quarter', 'unique', 'none', 'all'):
            for n in (3, 5, 10):
                for m in (5, 10, 25):
                    with self.subTest(family=family, n=n, m=m):
                        p, expected = fixture(n, m, family)
                        audit = exhaustive_audit(p)
                        self.assertEqual(audit['marked'], expected)
                        self.assertEqual(audit['audit_candidate_evaluations'], 2**n)
                        self.assertEqual(len(p.rules), m)

    def test_explicit_example(self):
        p = load_problem(Path(__file__).with_name('example.json'))
        self.assertEqual([x for x in range(16) if p.feasible(x)], [6])
        self.assertEqual([(a.employee, a.shift) for a in p.decode(6)],
                         [('Ben', 'S0'), ('Ada', 'S1')])

    def test_each_constraint_semantics(self):
        rules = [Rule('p', 'pattern', (0, 1), {'pattern': [1, 0]}),
                 Rule('c', 'count', (0, 1), {'lower': 1, 'upper': 1}),
                 Rule('w', 'weighted', (0, 1), {'lower': 2, 'upper': 4, 'weights': [2, 3]}),
                 Rule('e', 'equal', (0, 1), {})]
        expected = [[True, True, False, True], [False, True, True, False],
                    [False, True, True, False], [True, False, False, True]]
        patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for r, values in zip(rules, expected):
            self.assertEqual([r.accepts(row) for row in patterns], values)

    def test_counts_and_short_circuit(self):
        p = load_problem(Path(__file__).with_name('example.json'))
        c = Counts()
        self.assertFalse(p.feasible(1, c))
        self.assertEqual((c.candidates, c.constraints), (1, 1))
        c = Counts()
        self.assertFalse(p.feasible(1, c, short_circuit=False))
        self.assertEqual(c.constraints, 5)
        self.assertEqual(sum(c.by_rule.values()), c.constraints)

    def test_classical_search(self):
        p, _ = fixture(3, 5, 'none')
        row = classical_search(p, np.random.default_rng(2), budget=2)
        self.assertEqual(row['status'], 'budget_exhausted')
        self.assertEqual(row['candidate_evaluations'], 2)
        row = classical_search(p, np.random.default_rng(2), exhaustive=True)
        self.assertEqual(row['status'], 'proven_unsat')
        self.assertEqual(row['candidate_evaluations'], 8)
        p = load_problem(Path(__file__).with_name('example.json'))
        row = classical_search(p, np.random.default_rng(1), exhaustive=True)
        self.assertEqual((row['candidate'], row['candidate_evaluations']), (6, 7))

    def test_no_rules(self):
        p, _ = fixture(2, 1)
        p = Problem(p.assignments, [])
        self.assertEqual(exhaustive_audit(p)['marked'], 4)
        zero, expected = fixture(2, 0)
        self.assertEqual(expected, 4)
        self.assertEqual(zero.rules, [])

    def test_invalid_limits(self):
        for n, m in [(0, 5), (11, 5), (3, 26), (3, -1), (True, 5)]:
            with self.subTest(n=n, m=m), self.assertRaises(ValueError):
                fixture(n, m)

    def test_invalid_rules(self):
        cases = [('r', 'custom', (0,), {}), ('r', 'equal', (0,), {}),
                 ('r', 'count', (0,), {'lower': 2, 'upper': 1}),
                 ('r', 'pattern', (0,), {'pattern': [False]}),
                 ('r', 'weighted', (0,), {'lower': 0, 'upper': 2, 'weights': [-1]}),
                 ('r', 'count', (0, 0), {'lower': 0, 'upper': 1})]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Rule(*args)

    def test_invalid_problem(self):
        p, _ = fixture(3, 5)
        for rules in ([Rule('bad', 'equal', (0, 3), {})], [p.rules[0], p.rules[0]]):
            with self.assertRaises(ValueError):
                Problem(p.assignments, rules)
        with self.assertRaises(ValueError):
            Problem([p.assignments[0]]*2, [])

    def test_iterations(self):
        self.assertEqual(known_iterations(3, 2), 1)
        self.assertEqual(known_iterations(10, 1), 25)
        self.assertEqual(known_iterations(3, 8), 0)
        with self.assertRaises(ValueError):
            known_iterations(3, 9)

    def test_reproducible_classical_random(self):
        p, _ = fixture(5, 5)
        rows = [classical_search(p, np.random.default_rng(3), budget=5) for _ in range(2)]
        for key in ('candidate', 'candidate_evaluations', 'constraint_evaluations', 'success'):
            self.assertEqual(rows[0][key], rows[1][key])


def write_fixture_manifest():
    rows = []
    for family in ('quarter', 'unique', 'none', 'all'):
        for n in (3, 5, 10):
            for m in (5, 10, 25):
                p, expected = fixture(n, m, family)
                audit = exhaustive_audit(p)
                rows.append(dict(family=family, data_qubits=n, active_constraints=m,
                                 N=2**n, expected_M=expected, verified_M=audit['marked']))
    path = Path(__file__).with_name('verified_fixture_counts.csv')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClassicalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        write_fixture_manifest()
    raise SystemExit(not result.wasSuccessful())
