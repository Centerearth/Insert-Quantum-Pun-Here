"""Small, explicit scheduling predicates and a reversible Grover oracle.

x_i = 1 means assignment i is selected. Qubit i is the 2**i bit of x.
No coverage, availability, or other business policy is implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import asin, ceil, floor, log2, pi, sin, sqrt
from time import perf_counter
from typing import Any
import json
import numpy as np

MAX_QUBITS, MAX_CONSTRAINTS = 10, 25


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class Assignment:
    employee: str
    shift: str
    location: str
    role: str
    duration_units: int

    def __post_init__(self):
        for name in ('employee', 'shift', 'location', 'role'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        integer(self.duration_units, 'duration_units', 1, 10**9)


@dataclass(frozen=True)
class Rule:
    """Declarative interface; there is deliberately no arbitrary callable field.

    pattern: forbid exactly one local bit pattern.
    count: lower <= sum(local bits) <= upper.
    weighted: lower <= sum(weight_i * bit_i) <= upper (integer units).
    equal: two assignment bits must be equal.
    """
    name: str
    kind: str
    bits: tuple[int, ...]
    params: dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('rule name must be nonempty')
        if not isinstance(self.bits, (tuple, list)) or not self.bits:
            raise ValueError('rule bits must be a nonempty list/tuple')
        if any(type(i) is not int or i < 0 for i in self.bits):
            raise ValueError('bit indices must be nonnegative integers')
        if len(set(self.bits)) != len(self.bits):
            raise ValueError('duplicate bits in a rule')
        expected = {'pattern': {'pattern'}, 'count': {'lower', 'upper'},
                    'weighted': {'lower', 'upper', 'weights'}, 'equal': set()}
        if self.kind not in expected:
            raise ValueError(f'unsupported constraint type: {self.kind!r}')
        if not isinstance(self.params, dict) or set(self.params) != expected[self.kind]:
            raise ValueError(f'{self.kind} requires exactly {expected[self.kind]}')
        if self.kind == 'pattern':
            pattern = self.params['pattern']
            if not isinstance(pattern, (list, tuple)) or len(pattern) != len(self.bits):
                raise ValueError('pattern length must match bits')
            for bit in pattern:
                integer(bit, 'pattern bit', 0, 1)
        if self.kind in ('count', 'weighted'):
            lo, hi = self.params['lower'], self.params['upper']
            integer(lo, 'lower', 0, 10**12)
            integer(hi, 'upper', lo, 10**12)
        if self.kind == 'weighted':
            weights = self.params['weights']
            if not isinstance(weights, (list, tuple)) or len(weights) != len(self.bits):
                raise ValueError('weights length must match bits')
            for weight in weights:
                integer(weight, 'weight', 0, 10**9)
        if self.kind == 'equal' and len(self.bits) != 2:
            raise ValueError('equal requires two bits')

    def accepts(self, local_bits):
        """Exactly one deterministic yes/no constraint evaluation."""
        if self.kind == 'pattern':
            return tuple(local_bits) != tuple(self.params['pattern'])
        if self.kind == 'count':
            total = sum(local_bits)
        elif self.kind == 'weighted':
            total = sum(w*b for w, b in zip(self.params['weights'], local_bits))
        elif self.kind == 'equal':
            return local_bits[0] == local_bits[1]
        else:
            raise ValueError(f'unsupported constraint type: {self.kind}')
        return self.params['lower'] <= total <= self.params['upper']


@dataclass
class Counts:
    candidates: int = 0
    constraints: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)


@dataclass
class Problem:
    assignments: list[Assignment]
    rules: list[Rule]  # Only active rules go here; order determines short circuiting.
    name: str = 'custom'

    def __post_init__(self):
        integer(len(self.assignments), 'data qubits', 1, MAX_QUBITS)
        integer(len(self.rules), 'active constraints', 0, MAX_CONSTRAINTS)
        if not all(isinstance(a, Assignment) for a in self.assignments):
            raise ValueError('assignments must contain Assignment objects')
        if not all(isinstance(r, Rule) for r in self.rules):
            raise ValueError('rules must contain Rule objects')
        keys = [(a.employee, a.shift, a.location, a.role) for a in self.assignments]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate assignment options')
        if len({r.name for r in self.rules}) != len(self.rules):
            raise ValueError('constraint names must be unique')
        if any(i >= self.n for r in self.rules for i in r.bits):
            raise ValueError('constraint refers to an out-of-range data bit')

    @property
    def n(self):
        return len(self.assignments)

    def feasible(self, x, counts=None, short_circuit=True):
        integer(x, 'candidate', 0, 2**self.n - 1)
        if counts is not None:
            counts.candidates += 1
        good = True
        for rule in self.rules:
            if counts is not None:
                counts.constraints += 1
                counts.by_rule[rule.name] = counts.by_rule.get(rule.name, 0) + 1
            passed = rule.accepts(tuple((x >> i) & 1 for i in rule.bits))
            good = good and passed
            if not passed and short_circuit:
                break
        return good

    def decode(self, x):
        integer(x, 'candidate', 0, 2**self.n-1)
        return [a for i, a in enumerate(self.assignments) if (x >> i) & 1]


def load_problem(path):
    """JSON parameters are mandatory; unknown fields/types are rejected."""
    with open(path) as handle:
        data = json.load(handle)
    if set(data) != {'name', 'assignments', 'constraints'}:
        raise ValueError('JSON requires exactly name, assignments, constraints')
    return Problem([Assignment(**a) for a in data['assignments']],
                   [Rule(**r) for r in data['constraints']], data['name'])


def toggle(qc, controls, target):
    if controls:
        qc.mcx(list(controls), target)
    else:
        qc.x(target)


def phase_all_ones(qc, bits):
    if len(bits) == 1:
        qc.z(bits[0])
    else:
        qc.h(bits[-1])
        toggle(qc, bits[:-1], bits[-1])
        qc.h(bits[-1])


def compile_rule(rule):
    """Explicit bounded truth-table synthesis, NOT automatic Python compilation.

    Return V: |local x>|b> -> |local x>|b XOR violation(x)>.
    Enumerates only this rule's support. This is exponential in support size!
    Choosing the smaller truth-table half avoids huge circuits for sparse rules.
    """
    from qiskit import QuantumCircuit
    integer(len(rule.bits), 'rule support size', 1, MAX_QUBITS)
    rows = list(product((0, 1), repeat=len(rule.bits)))
    valid = [row for row in rows if rule.accepts(row)]
    valid_set = set(valid)
    invalid = [row for row in rows if row not in valid_set]
    qc = QuantumCircuit(len(rule.bits)+1, name=rule.name)
    # If valid rows are fewer: violation = 1 XOR membership(valid).
    complement = len(valid) < len(invalid)
    patterns = valid if complement else invalid
    target = len(rule.bits)
    if complement:
        qc.x(target)
    for row in patterns:
        zeros = [i for i, bit in enumerate(row) if bit == 0]
        for i in zeros:
            qc.x(i)
        toggle(qc, list(range(target)), target)
        for i in reversed(zeros):
            qc.x(i)
    return qc.to_gate(), len(rows), len(patterns)


@dataclass
class Oracle:
    circuit: QuantumCircuit
    compile_seconds: float
    local_table_evaluations: int
    minterms: int
    ancillas: int


def build_oracle(problem):
    """Mark iff violation count is zero; uncompute ALL work qubits.

    A b-bit counter holds 0..m, b=ceil(log2(m+1)). No overflow for m rules.
    One flag is reused. Each rule block is used four times per phase oracle:
    compute flag, increment, clear flag, then inverse of that whole computation.
    """
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
    start = perf_counter()
    n, m = problem.n, len(problem.rules)
    data = QuantumRegister(n, 'assignment')
    if m == 0:
        qc = QuantumCircuit(data, name='all_feasible')
        qc.global_phase = pi
        return Oracle(qc, perf_counter()-start, 0, 0, 0)
    b = ceil(log2(m+1))
    work = AncillaRegister(b+1, 'work')
    compute = QuantumCircuit(data, work)
    counter, flag = list(work[:b]), work[b]
    evaluations = minterms = 0
    for rule in problem.rules:
        gate, calls, terms = compile_rule(rule)
        evaluations += calls
        minterms += terms
        args = [data[i] for i in rule.bits] + [flag]
        compute.append(gate, args)
        # Controlled +1 modulo 2**b: high bits first, using old lower bits.
        for j in range(b-1, 0, -1):
            toggle(compute, [flag] + counter[:j], counter[j])
        compute.cx(flag, counter[0])
        compute.append(gate.inverse(), args)
    qc = QuantumCircuit(data, work, name='feasibility_phase')
    qc.compose(compute, inplace=True)
    qc.x(counter)
    phase_all_ones(qc, counter)
    qc.x(counter)
    qc.compose(compute.inverse(), inplace=True)
    return Oracle(qc, perf_counter()-start, evaluations, minterms, b+1)


def grover_circuit(problem, oracle, iterations):
    integer(iterations, 'iterations', 0, 10000)
    qc = oracle.circuit.copy_empty_like()
    qc.global_phase = 0
    data = list(range(problem.n))
    qc.h(data)
    for _ in range(iterations):
        qc.compose(oracle.circuit, inplace=True)
        # Diffusion on data ONLY. Work qubits are back to |0>.
        qc.h(data)
        qc.x(data)
        phase_all_ones(qc, data)  # Equivalent diffuser up to global phase.
        qc.x(data)
        qc.h(data)
    qc.measure_all()  # Work measurements also let tests detect dirty ancillas.
    return qc


def known_iterations(n, marked):
    """Use supplied benchmark knowledge; never silently estimate M for free."""
    integer(marked, 'marked count', 0, 2**n)
    if marked == 0:
        # Deliberately execute a stress run; failure is not an UNSAT proof.
        return max(1, floor(pi/4 * sqrt(2**n)))
    if marked == 2**n:
        return 0
    theta = asin(sqrt(marked / 2**n))
    # Best of floor/ceil around first peak; ties prefer fewer calls.
    peak = pi/(4*theta)-0.5
    choices = sorted({max(0, floor(peak)), max(0, ceil(peak))})
    return max(choices, key=lambda k: (round(sin((2*k+1)*theta)**2, 14), -k))


class GroverEngine:
    """One shot per attempt; no hidden shot batching or sample reuse."""
    def __init__(self, problem):
        from qiskit_aer import AerSimulator
        self.problem = problem
        self.oracle = build_oracle(problem)
        self.backend = AerSimulator(method='statevector', max_parallel_threads=1)
        self.cache = {}

    def prepare(self, iterations):
        from qiskit import transpile
        if iterations in self.cache:
            return self.cache[iterations], 0.0
        start = perf_counter()
        # Explicit basis makes size comparable and introduces no new ancillas.
        qc = transpile(grover_circuit(self.problem, self.oracle, iterations),
                       basis_gates=['u', 'cx'], optimization_level=0,
                       seed_transpiler=0)
        self.cache[iterations] = qc
        return qc, perf_counter()-start

    def run(self, rng, *, marked=None, attempts=1, iteration_plan=None):
        integer(attempts, 'attempts', 1, 1000)
        if iteration_plan is None:
            iteration_plan = ([known_iterations(self.problem.n, marked)]*attempts
                              if marked is not None else
                              [int(rng.integers(0, ceil(sqrt(2**self.problem.n))))
                               for _ in range(attempts)])
        if len(iteration_plan) != attempts:
            raise ValueError('iteration_plan length must match attempts')
        for k in iteration_plan:
            integer(k, 'planned iterations', 0, 10000)
        counts = Counts()
        ks, samples = [], []
        sim_seconds = transpile_seconds = 0.0
        cx_total = gates_total = max_depth = 0
        start = perf_counter()
        success = False
        for k in iteration_plan:
            qc, compile_time = self.prepare(k)
            transpile_seconds += compile_time
            t = perf_counter()
            result = self.backend.run(qc, shots=1, memory=True,
                                      seed_simulator=int(rng.integers(0, 2**31))).result()
            sim_seconds += perf_counter()-t
            measured = int(result.get_memory()[0].replace(' ', ''), 2)
            if measured >> self.problem.n:
                raise RuntimeError('oracle left nonzero work qubits')
            samples.append(measured)
            ks.append(k)
            cx_total += qc.count_ops().get('cx', 0)
            gates_total += qc.size()
            max_depth = max(max_depth, qc.depth())
            success = self.problem.feasible(measured, counts)
            if success:
                break
        elapsed = perf_counter()-start
        calls = sum(ks)
        return dict(method='grover', success=success,
                    status='found' if success else 'no_verified_sample',
                    candidate=measured if success else None,
                    sample_integers=json.dumps(samples),
                    iterations=json.dumps(ks), grover_iterations=calls,
                    logical_oracle_calls=calls,
                    candidate_evaluations=counts.candidates,
                    constraint_evaluations=counts.constraints,
                    constraint_evaluations_by_rule=json.dumps(counts.by_rule),
                    reversible_rule_blocks=4*len(self.problem.rules)*calls,
                    total_decision_queries=calls+counts.candidates,
                    simulator_seconds=sim_seconds,
                    transpile_seconds=transpile_seconds,
                    search_wall_seconds=elapsed,
                    executed_cx=cx_total, executed_gates=gates_total,
                    max_circuit_depth=max_depth,
                    circuit_qubits=self.oracle.circuit.num_qubits,
                    ancillas=self.oracle.ancillas,
                    attempts_used=len(ks), iteration_policy='known_M' if marked is not None else 'bounded_random_length')


def classical_search(problem, rng, budget=None, exhaustive=False):
    """Random permutation (no replacement) or lexicographic exhaustive search."""
    N = 2**problem.n
    if budget is not None:
        integer(budget, 'budget', 1, 10**9)
    start = perf_counter()
    order = range(N) if exhaustive else rng.permutation(N)
    limit = N if budget is None else min(N, budget)
    counts = Counts()
    found = None
    for x in order:
        if counts.candidates >= limit:
            break
        if problem.feasible(int(x), counts):
            found = int(x)
            break
    elapsed = perf_counter()-start
    exhausted = counts.candidates == N and found is None
    return dict(method='exhaustive_first' if exhaustive else 'random_without_replacement',
                success=found is not None,
                status='found' if found is not None else ('proven_unsat' if exhausted else 'budget_exhausted'),
                candidate=found, grover_iterations=0, logical_oracle_calls=0,
                candidate_evaluations=counts.candidates,
                constraint_evaluations=counts.constraints,
                constraint_evaluations_by_rule=json.dumps(counts.by_rule),
                reversible_rule_blocks=0, total_decision_queries=counts.candidates,
                simulator_seconds=0.0, transpile_seconds=0.0,
                search_wall_seconds=elapsed, ancillas=0, circuit_qubits=0)


def exhaustive_audit(problem):
    counts = Counts()
    start = perf_counter()
    marked = sum(problem.feasible(x, counts) for x in range(2**problem.n))
    return dict(marked=marked, audit_candidate_evaluations=counts.candidates,
                audit_constraint_evaluations=counts.constraints,
                audit_seconds=perf_counter()-start,
                feasibility='proven_unsat' if marked == 0 else 'feasible')


def fixture(n, m, family='quarter'):
    """Explicit synthetic policies, not inferred real business requirements.

    Repeated rules deliberately test m independently of solution density.
    Every assignment is one synthetic employee E0 on a separate unit shift.
    """
    integer(n, 'data qubits', 1, MAX_QUBITS)
    integer(m, 'active constraints', 0, MAX_CONSTRAINTS)
    assignments = [Assignment('E0', f'S{i}', 'L0', 'R0', 1) for i in range(n)]
    if family not in {'quarter', 'unique', 'none', 'all'}:
        raise ValueError(f'unknown fixture family: {family}')
    if m == 0:
        return Problem(assignments, [], f'{family}_{n}q_0c'), 2**n
    if family == 'quarter':
        # First min(2,n,m) shifts unavailable. Remaining bits are unrestricted.
        fixed = min(2, n, m)
        rules = [Rule(f'unavailable_{j}', 'pattern', (j,), {'pattern': [1]})
                 for j in range(fixed)]
        expected = 2**(n-fixed)
    elif family == 'unique':
        rules = [Rule('zero_shift_count', 'count', tuple(range(n)), {'lower': 0, 'upper': 0})]
        expected = 1
    elif family == 'none':
        # Impossible for one bit, but a valid explicit hard constraint.
        rules = [Rule('impossible_count', 'count', (0,), {'lower': 2, 'upper': 2})]
        expected = 0
    elif family == 'all':
        rules = [Rule('unrestricted_count', 'count', (0,), {'lower': 0, 'upper': 1})]
        expected = 2**n
    else:
        raise ValueError(f'unknown fixture family: {family}')
    while len(rules) < m:
        source = rules[(len(rules)-1) % len(rules)]
        rules.append(Rule(f'redundant_{len(rules)}', source.kind, source.bits, dict(source.params)))
    return Problem(assignments, rules, f'{family}_{n}q_{m}c'), expected
