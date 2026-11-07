"""Classical baselines that run on EXACTLY the same rules as the quantum oracle.

1. brute_force_random  - check whole schedules in random order (what Grover is compared to)
2. backtracking        - fill in one yes/no decision at a time and abandon a branch the
                         moment a rule can no longer be satisfied (simple "smart" search)
3. cpsat               - Google OR-Tools CP-SAT, an industrial constraint solver
4. real_size_cpsat     - CP-SAT on a realistic facility week (far beyond any quantum simulator)
"""
from time import perf_counter
import random
from ortools.sat.python import cp_model


# ---------- 1. Brute force -------------------------------------------------------------
def brute_force_random(problem, rng):
    """Check full schedules in a random order until one passes. Returns checks used."""
    order = list(range(2 ** problem.n))
    rng.shuffle(order)
    for checks, x in enumerate(order, start=1):
        if problem.feasible(x):
            return checks
    return len(order)


# ---------- 2. Backtracking --------------------------------------------------------------
def rule_still_possible(rule, partial):
    """partial maps bit -> 0/1 for the bits decided so far. Can this rule still pass?"""
    vals = [partial.get(b) for b in rule.bits]
    if rule.kind == 'pattern':
        # Violated only once every bit is decided AND they match the forbidden pattern.
        return any(v is None or v != p for v, p in zip(vals, rule.params['pattern']))
    if rule.kind == 'equal':
        return None in vals or vals[0] == vals[1]
    weights = rule.params.get('weights', [1] * len(rule.bits))
    lowest = sum(w for w, v in zip(weights, vals) if v == 1)          # all undecided -> 0
    highest = lowest + sum(w for w, v in zip(weights, vals) if v is None)  # all undecided -> 1
    return lowest <= rule.params['upper'] and highest >= rule.params['lower']


def backtracking(problem, rng):
    """Depth-first search over the bits. Returns (solution, nodes visited).

    A node = one partial schedule we looked at. The value order (try 0 or 1 first)
    is randomized so results are not an artifact of a lucky ordering."""
    nodes = 0

    def search(i, partial):
        nonlocal nodes
        nodes += 1
        if not all(rule_still_possible(r, partial) for r in problem.rules):
            return None                      # prune: this branch can never work
        if i == problem.n:
            return sum(v << b for b, v in partial.items())
        for value in rng.sample([0, 1], 2):
            partial[i] = value
            found = search(i + 1, partial)
            if found is not None:
                return found
            del partial[i]
        return None

    solution = search(0, {})
    assert solution is None or problem.feasible(solution)
    return solution, nodes


# ---------- 3. CP-SAT on the same toy instance ------------------------------------------
def add_rule(model, rule, x):
    v = [x[b] for b in rule.bits]
    if rule.kind == 'pattern':
        # Forbid the pattern: at least one bit differs from it.
        model.AddBoolOr([lit.Not() if p == 1 else lit for lit, p in zip(v, rule.params['pattern'])])
    elif rule.kind == 'equal':
        model.Add(v[0] == v[1])
    else:
        w = rule.params.get('weights', [1] * len(v))
        total = sum(wi * vi for wi, vi in zip(w, v))
        model.Add(total >= rule.params['lower'])
        model.Add(total <= rule.params['upper'])


def cpsat(problem, seed=0):
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x{i}') for i in range(problem.n)]
    for rule in problem.rules:
        add_rule(model, rule, x)
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    sol = sum(int(solver.Value(xi)) << i for i, xi in enumerate(x)) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    assert sol is None or problem.feasible(sol)
    return dict(status=solver.StatusName(status), solution=sol, branches=solver.NumBranches(),
                conflicts=solver.NumConflicts(), wall_seconds=solver.WallTime())


# ---------- 4. CP-SAT at real size -------------------------------------------------------
def real_size_cpsat(num_staff=30, days=7, shifts=3, seed=7, optimize=True, time_limit=60):
    """A realistic facility week. Same KINDS of rules as the toy, plus the counting
    rules a real facility needs. Returns size and solve statistics."""
    rnd = random.Random(seed)
    S = range(num_staff); D = range(days); H = range(shifts)        # H: 0 day, 1 evening, 2 night
    is_rn = [i < num_staff // 3 for i in S]
    cost = [rnd.randint(45, 60) if is_rn[i] else rnd.randint(25, 35) for i in S]
    need = [max(2, num_staff // 6), max(2, num_staff // 8), max(1, num_staff // 10)]
    unavailable = {(i, d, h) for i in S for d in D for h in H if rnd.random() < 0.10}
    conflicts = [tuple(rnd.sample(list(S), 2)) for _ in range(max(1, num_staff // 6))]
    prefers = {(i, d, h): rnd.randint(0, 3) for i in S for d in D for h in H}

    m = cp_model.CpModel()
    x = {(i, d, h): m.NewBoolVar(f'x{i}_{d}_{h}') for i in S for d in D for h in H}
    for d in D:
        for h in H:
            m.Add(sum(x[i, d, h] for i in S) >= need[h])                    # coverage
            m.Add(sum(x[i, d, h] for i in S) <= need[h] + 2)                # don't overstaff
            m.Add(sum(x[i, d, h] for i in S if is_rn[i]) >= 1)              # RN every shift
            for a, b in conflicts:
                m.AddBoolOr([x[a, d, h].Not(), x[b, d, h].Not()])          # personnel conflicts
    for i in S:
        for d in D:
            m.Add(sum(x[i, d, h] for h in H) <= 1)                          # no doubles
            if d + 1 < days:
                m.AddBoolOr([x[i, d, 2].Not(), x[i, d + 1, 0].Not()])      # rest after nights
        m.Add(sum(x[i, d, h] for d in D for h in H) <= 5)                   # max 5 shifts/week
    for key in unavailable:
        m.Add(x[key] == 0)                                                  # availability
    weekly_cost = sum(8 * cost[i] * x[i, d, h] for i in S for d in D for h in H)
    if optimize:
        # Best schedule: honour preferences, minus a small weight on payroll.
        m.Maximize(sum(prefers[k] * x[k] for k in x) * 100 - weekly_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    start = perf_counter()
    status = solver.Solve(m)
    return dict(staff=num_staff, days=days, shifts=shifts, binary_variables=len(x),
                schedules_log2=len(x), constraints=len(m.Proto().constraints),
                optimize=optimize, status=solver.StatusName(status),
                wall_seconds=round(perf_counter() - start, 3),
                branches=solver.NumBranches(), conflicts=solver.NumConflicts(),
                objective=(solver.ObjectiveValue() if optimize and status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None),
                gap_to_proven_best_percent=(None if not optimize or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                    else round(100 * abs(solver.ObjectiveValue() - solver.BestObjectiveBound())
                               / max(1.0, abs(solver.BestObjectiveBound())), 3)))
