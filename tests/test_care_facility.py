"""Checks for the care-facility instances and classical baselines.   python -m pytest -q"""
import random
import numpy as np
import pytest
from instances import INSTANCES, care_facility, describe
from classical import backtracking, brute_force_random, cpsat
from test_scheduling import state_after

ALL_SIZES = sorted(INSTANCES)


def solutions(p):
    return {x for x in range(2 ** p.n) if p.feasible(x)}


def test_expected_solution_counts():
    assert [len(solutions(INSTANCES[n]())) for n in ALL_SIZES] == [2, 2, 3]


def test_every_solution_obeys_the_business_rules_in_plain_terms():
    for n in ALL_SIZES:
        p = INSTANCES[n]()
        for x in solutions(p):
            picked = [(a.employee, a.shift) for a in p.decode(x)]
            people = [e for e, _ in picked]
            assert len(people) == len(set(people)), 'nobody works a double'
            assert any(e in ('Ana', 'Cara') for e, s in picked if s == 'Day'), 'RN on Day'
            assert any(e in ('Ana', 'Cara') for e, s in picked if s == 'Night'), 'RN on Night'
            assert ('Ben', 'Night') not in picked
            assert not {('Ana', 'Day'), ('Dev', 'Day')} <= set(picked)


@pytest.mark.parametrize('n', ALL_SIZES)
def test_oracle_marks_exactly_the_valid_schedules(n):
    # Phase oracle on the uniform superposition: valid schedules get a minus sign,
    # everything else unchanged, and all work qubits return to 0.
    p = INSTANCES[n]()
    v = state_after(p)
    target = np.zeros_like(v)
    good = solutions(p)
    for x in range(2 ** n):
        target[x] = (-1 if x in good else 1) / np.sqrt(2 ** n)
    np.testing.assert_allclose(v, target, atol=1e-9)


@pytest.mark.parametrize('n', ALL_SIZES)
def test_classical_solvers_return_valid_schedules(n):
    p = INSTANCES[n](); good = solutions(p); rng = random.Random(0)
    for _ in range(50):
        sol, nodes = backtracking(p, rng)
        assert sol in good and nodes >= 1
        assert 1 <= brute_force_random(p, rng) <= 2 ** n
    assert cpsat(p)['solution'] in good


def test_describe_is_readable():
    p = care_facility(2)
    assert describe(p, 0) == '(nobody)'
    assert 'Ana->Day' in describe(p, 1)
