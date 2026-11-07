"""Realistic care-facility scheduling instances for the experiment.

Client story: a scheduling outsourcer builds weekly staff schedules for care
facilities. Each instance below is ONE day at one facility: a Day shift and a
Night shift, and a small team of staff.

Encoding (one qubit per yes/no decision):
    bit = 2*staff_index + shift_index        shift 0 = Day, shift 1 = Night
    bit value 1 -> "this person works this shift"

So 2 staff -> 4 bits (16 schedules), 3 staff -> 6 bits (64), 4 staff -> 8 bits (256).

Every rule is a real scheduling rule a facility would actually state, written
with the four rule types the oracle understands (pattern, count, weighted, equal).
"""
from scheduling import Assignment, Problem, Rule

DAY, NIGHT = 0, 1
SHIFT_NAMES = ['Day', 'Night']

# name, is a registered nurse (RN)?, hourly cost in dollars
STAFF = [('Ana', True, 52), ('Cara', True, 48), ('Ben', False, 30), ('Dev', False, 28)]


def bit(staff_index, shift):
    return 2 * staff_index + shift


def care_facility(num_staff):
    """Build the instance for 2, 3 or 4 staff. Rules are listed in plain English."""
    staff = STAFF[:num_staff]
    assignments = [Assignment(name, SHIFT_NAMES[s], 'Maple Grove', 'RN' if rn else 'Aide', 12)
                   for name, rn, _ in staff for s in (DAY, NIGHT)]
    everyone = range(num_staff)
    rns = [i for i, (_, rn, _) in enumerate(staff) if rn]
    rules = []

    # 1-2. Coverage: each shift needs a minimum (and maximum) headcount.
    day_need = (1, 1) if num_staff == 2 else (2, 2)
    rules.append(Rule('day_headcount', 'count', tuple(bit(i, DAY) for i in everyone),
                      {'lower': day_need[0], 'upper': day_need[1]}))
    rules.append(Rule('night_headcount', 'count', tuple(bit(i, NIGHT) for i in everyone),
                      {'lower': 1, 'upper': 1}))

    # 3-4. A registered nurse on every shift (care regulation).
    for shift in (DAY, NIGHT):
        rules.append(Rule(f'rn_on_{SHIFT_NAMES[shift].lower()}', 'count',
                          tuple(bit(i, shift) for i in rns), {'lower': 1, 'upper': len(rns)}))

    # 5+. Nobody works a double (Day AND Night = 24 hours).
    for i in everyone:
        rules.append(Rule(f'no_double_{staff[i][0]}', 'pattern',
                          (bit(i, DAY), bit(i, NIGHT)), {'pattern': [1, 1]}))

    # Availability: Ben can't work nights (childcare).
    if num_staff >= 3:
        rules.append(Rule('Ben_unavailable_night', 'pattern', (bit(2, NIGHT),), {'pattern': [1]}))

    # Personnel conflict: Ana and Dev can't be on the same Day shift.
    if num_staff >= 4:
        rules.append(Rule('Ana_Dev_not_together_day', 'pattern',
                          (bit(0, DAY), bit(3, DAY)), {'pattern': [1, 1]}))

    # Budget: total hourly cost of the Day shift at most $85/hour.
    if num_staff >= 3:
        day_bits = tuple(bit(i, DAY) for i in everyone)
        rules.append(Rule('day_budget_85_per_hour', 'weighted', day_bits,
                          {'weights': [staff[i][2] for i in everyone], 'lower': 0, 'upper': 85}))

    return Problem(assignments, rules, f'care_facility_{num_staff}_staff')


def describe(problem, x):
    """Turn a bitstring into a readable schedule."""
    picked = problem.decode(x)
    return ', '.join(f'{a.employee}->{a.shift}' for a in picked) or '(nobody)'


INSTANCES = {4: lambda: care_facility(2), 6: lambda: care_facility(3), 8: lambda: care_facility(4)}
