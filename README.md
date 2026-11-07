# Employee schedule feasibility with Grover search

This is a small educational implementation for **1–10 data qubits** and **0–25 active hard constraints**. The default grid uses 3, 5, 10 data qubits × 5, 10, 25 constraints. It targets Qiskit 2.x and Qiskit Aer, using a noiseless CPU statevector simulator. It does not require a qBraid cloud-device credential or send jobs to quantum hardware.

**Validation status:** Python syntax checks and all 11 classical test methods passed in the creation environment, including exhaustive counts for 36 fixture configurations. Qiskit and Aer were absent, and the environment denied package installation with HTTP 403. Therefore the Qiskit/Aer tests and timed quantum benchmarks have **not been executed here**. `test_scheduling.py` supplies phase, clean-work, Grover, and boundary checks to run after installation. No simulated performance results are fabricated.

## Run in qBraid Lab

Extract the archive into your project and open its `grover_scheduling` directory in the terminal. Use a Python 3.10+ environment; dependencies may already exist in your qBraid environment. Install into the same interpreter used by your notebook/kernel:

```bash
python -m pip install -r requirements.txt
python validate_classical.py
python -m pytest -q
python benchmark.py --trials 20 --seed 2026
```

In a notebook use `%pip install -r requirements.txt`, then restart the kernel if necessary. `%run benchmark.py --trials 20 --seed 2026` runs the experiment there. When using a different working directory, use the full path or change into this directory first.

Other examples:

```bash
# A different grid within the bounds
python benchmark.py --qubits 4 6 --constraints 3 12 --trials 30 --out results_custom

# Low-density case: M=1 (larger circuits; start with small settings)
python benchmark.py --family unique --qubits 3 5 --constraints 5 --trials 10

# Small UNSAT and all-feasible controls
python benchmark.py --family none --qubits 3 --constraints 5 --trials 5
python benchmark.py --family all --qubits 3 --constraints 5 --trials 5

# Bounded search without disclosing M to the quantum algorithm
python benchmark.py --unknown-m --attempts 4 --qubits 3 --constraints 5

# Explicit employee/shift example; JSON supplies its own qubit/constraint counts
python benchmark.py --problem example.json --trials 20 --out example_results
```

The default is one quantum attempt per trial, with one shot per attempt. Each attempt really executes a fresh Aer job. `--attempts` gives a maximum number of attempts; search stops after the first verified feasible sample. Failed samples are retained in `trials.csv`. The seed controls NumPy random streams, simulator seeds, and fixed transpilation seeds. Reproducibility does not imply identical timing across machines or package versions.

## Encoding and explicit assumptions

A data bit represents one **assignment option**, not an entire schedule:

- `x_i = 1`: select the listed `(employee, shift, location, role)` assignment.
- `x_i = 0`: do not select it.
- An n-bit basis state is one complete choice among **N = 2^n** candidate schedules.
- All N bitstrings are candidates, including empty schedules and conflicting choices. They are rejected only by explicitly active rules.
- Assignment metadata identifies employees and shifts and records a positive integer duration. The search engine does not infer time overlaps, employee skills, hours, coverage requirements, or eligibility from that metadata.

You explicitly list the options being modeled. Leaving an option out excludes it from the search space; a meaningful classical comparison must use that same space. Ten bits support ten binary options, not ten employees times an arbitrary number of shifts.

Bit i is qubit i and the coefficient of 2^i in the candidate integer. Qiskit measurement strings display the highest-index qubit on the left. `decode(x)` returns the selected assignments and `int(memory_string, 2)` handles this ordering. Work bits occupy higher indices than data bits and are checked to be zero after measurement.

To use real data, supply: the assignment list and durations/units; which rules are active; each rule's exact bit indices and parameters; and any external facts (availability, skills, start/end times, locations, wages) needed to derive those rules. There are no real business defaults.

## Working classical constraints and reversible implementations

`Rule(name, kind, bits, params)` is the interface. A problem's `rules` list contains only active rules. It may contain zero rules (every candidate is feasible), and at most 25. Names must be unique. List order controls classical short circuiting.

| Implemented kind | Mandatory parameters | Classical yes/no test | Reversible implementation |
|---|---|---|---|
| `pattern` | `pattern`: one 0/1 value for each listed bit | Local bits must differ from that forbidden pattern | Controlled-X minterms toggle a violation flag |
| `count` | Integer `lower`, `upper` | Inclusive bounds on sum of listed bits | Same bounded local truth-table synthesis |
| `weighted` | Nonnegative integer `weights`, integer `lower`, `upper` | Inclusive bounds on sum of `weight_i * bit_i` | Same bounded local truth-table synthesis |
| `equal` | Empty dictionary, exactly two bits | Both bits have equal values | Same synthesis, rejecting 01 and 10 |

Every call to `accepts(local_bits)` is one individual classical constraint evaluation. `Problem.feasible` extracts each rule's local bits, checks rules in order, and stops at the first violation by default. With `short_circuit=False`, it evaluates every rule. The constraints never assign scores or select a best match.

Common explicit encodings:

| Scheduling condition | Supported formulation |
|---|---|
| An assignment is unavailable or has a disallowed location/role | `pattern` on that bit, forbidden `[1]` |
| At least/exactly/at most k qualified staff on a shift | `count` over the explicitly identified qualified assignment bits, with the required bounds |
| A selected employee must be paired with another on that shift | `equal` on their selection bits; requires both or neither |
| Two employees cannot share a shift; or two assignments conflict | `pattern` on the two bits, forbidden `[1,1]` |
| One-way pairing implication x implies y | `pattern` on `[x,y]`, forbidden `[1,0]` |
| Employee shift-count bounds | `count` over that employee's listed options |
| Hours limit or an additive payroll cap | `weighted` with durations or integer currency units supplied explicitly |
| Binary continuity for a particular employee over two options | `equal`; larger continuity requirements need a correctly chosen set of such rules |

Coverage bit subsets must be derived from actual supplied skills/roles. This implementation does not independently validate certifications. Hours weights are explicit parameters; the code does not automatically copy or check them against assignment durations. Choose consistent units and ensure weights describe your intended business rule.

**Not implemented as general business-rule types:** automatic rest-time checks from timestamps, arbitrary consecutive-work rules, multi-employee continuity, overtime with thresholds or variable pay rates, weekend-count differences/fairness, qualification databases, and general policies/callables. A *specified finite forbidden combination* can use `pattern`; deriving the right combinations from calendar data requires custom modeling. Each generated `Rule` counts toward 25. Nonlinear overtime and count-difference fairness require a new rule type or a custom reversible circuit, not the nonnegative-weighted-sum interface.

### How the oracle actually works

An arbitrary Python checker cannot run on a superposition. Python branches and reads ordinary values; a quantum oracle needs a coherent unitary made of quantum gates, with all temporary information uncomputed.

Here the bridge is deliberately small and explicit. For each **supported declarative rule**, `compile_rule` enumerates its at-most-10-bit local input table. It constructs a gate

`V_j |local x>|b> = |local x>|b XOR violation_j(x)>`.

For a specified minterm, X gates convert zero controls to one controls, a multi-controlled X toggles the flag, and the X gates are undone. Distinct minterms are disjoint, so XOR of their indicators equals their Boolean union. The compiler chooses the smaller of the valid/invalid pattern sets; for valid patterns it starts with X on the flag, implementing `violation = 1 XOR valid`.

This is **truth-table synthesis**, not an efficient general-purpose reversible compiler. A k-bit rule costs 2^k classical predicate evaluations during compilation and potentially exponentially many minterm gates. The compiler does not enumerate *global feasible schedules* or use the exhaustive audit to construct the oracle. A full-support rule's local table nevertheless spans the full candidate space; that work is real and separately counted. This approach is useful for correctness and small examples, not evidence of end-to-end asymptotic advantage.

To combine m active rules without allocating m flags:

1. Allocate one reused flag and a `b = ceil(log2(m+1))`-bit violation counter.
2. For each rule, compute its violation flag, add that flag to the counter with controlled-X gates, then uncompute the flag.
3. Flip phase only when the counter is zero.
4. Apply the inverse of the entire counting computation.

The final map is `|x>|0...0> -> (-1)^feasible(x)|x>|0...0>`. The counter holds 0 through m without overflow. With m>0, there are **b+1 ancillas**: 4 for m=5, 5 for m=10, and 6 for m=25. At the maximum configuration there are **16 total circuit qubits**. Zero constraints use a global phase and no work qubits. Explicit transpilation to `u,cx` uses no additional ancillas; actual compiled qubit counts are recorded. No resets or intermediate measurements implement the oracle.

Each Grover iteration applies this complete phase oracle followed by a diffuser **on data qubits only**. All data qubits start in the uniform superposition. After measurement, the classical checker verifies the candidate; an invalid sample is a failure, never a best-match answer.

### Add a supported rule type

For a small truth-table-backed type, add its parameter schema and validation in `Rule.__post_init__`, and its deterministic local predicate in `Rule.accepts`. No change to `GroverEngine`, the counter construction, or classical search is needed. For example, a hard parity condition can be added with these three insertions:

```python
# 1. Add to the `expected` dictionary in Rule.__post_init__:
'parity': {'value'}

# 2. Add to validation in Rule.__post_init__:
if self.kind == 'parity':
    integer(self.params['value'], 'parity value', 0, 1)

# 3. Add BEFORE the other cases in Rule.accepts:
if self.kind == 'parity':
    return sum(local_bits) % 2 == self.params['value']
```

Then `Rule('even_selection', 'parity', (0,1,2), {'value': 0})` is supported by the same explicit bounded synthesizer. This is an extension example, **not enabled in the delivered parser**. Add classical truth-table and phase/work-cleanup tests before using it. For larger structured predicates, replace/extend `compile_rule` with a hand-designed reversible gate satisfying the same XOR-flag contract. If that gate needs extra clean scratch qubits, extend the allocation, gate mapping and reporting explicitly. Merely adding a classical function without a reversible implementation is insufficient.

## Controlled fixtures and supplied example

These fixtures contain deliberately chosen **synthetic**, explicitly configured policies; their numbers are not inferred business requirements. They use employee E0 with one unit-duration option per distinct synthetic shift S0…S(n-1), at L0 in role R0.

| Family | Explicit hard rules before padding | Known satisfying count M |
|---|---|---|
| `quarter` (default) | First min(2,n,m) assignment bits unavailable | `2^(n-min(2,n,m))` |
| `unique` | Total selected assignments exactly zero | 1 |
| `none` | A one-bit selection count must equal 2 | 0 |
| `all` | A one-bit selection count lies between 0 and 1 | `2^n` |

With m=0 every family has no active constraints and M=2^n.

Additional active constraints are uniquely named copies of existing rules, labeled `redundant_*`. They add circuit work without changing M. **Twenty-five active constraints do not imply 25 independent restrictions.** Default `quarter` counts are M=2,8,256 for n=3,5,10, respectively, for every default m. Their density stays 1/4, so one Grover iteration suffices; this family isolates constraint overhead and does not demonstrate scaling in N/M. Use `unique` to vary N/M, starting with small settings. Redundancy could be simplified away in a production system; this experiment intentionally retains it as a controlled cost factor.

`verified_fixture_counts.csv` contains exhaustive classical counts for all 36 default-axis × family cases; it is not a quantum benchmark result.

`example.json` provides two synthetic employees, Ada and Ben, and two four-unit shifts. It explicitly specifies five rules: Ada unavailable S0; exactly one operator per shift; Ben at most one shift; Ada at most four duration units. Its sole feasible integer is **6**, bits `(x0,x1,x2,x3)=(0,1,1,0)`: Ben works S0, Ada works S1. The four-unit durations and bounds are illustrative test inputs. No unstated rest, fairness, pay or preference rule applies.

## Benchmark design and metrics

Each problem gets an exhaustive **setup audit** over at most 1024 candidates. It records M, confirms the synthetic expected count when applicable, and establishes feasibility/UNSAT. This audit is not charged to either timed search, and its work/time is reported separately. If you include it in the overall workflow cost, it has already solved the small feasibility instance classically. This is an educational controlled benchmark, not a claim of practical speedup.

The default `known_M` policy receives that audited count. For `0<M<N`, it chooses the better integer around the first peak of

`P(success after k iterations) = sin²((2k+1) asin(sqrt(M/N)))`.

It uses zero iterations for M=N. For M=0 it deliberately runs a finite stress circuit, which cannot prove UNSAT. Known-M selection is benchmark assistance, not free knowledge available on a real search problem.

`--unknown-m` withholds the audit count from `GroverEngine.run`. Each attempt independently chooses an integer uniformly from `0..ceil(sqrt(N))-1`. This is a simple bounded randomized-length policy, **not a full BBHT algorithm**, amplitude estimation, or a complete unknown-M solver. A failed bounded run has no completeness guarantee.

The two classical comparisons use identical n-bit candidates and `Problem.feasible` rules:

- **Randomized unstructured search:** a uniform random permutation, without replacement, stops on the first feasible candidate or its candidate cap. For t inspected candidates its ideal success probability is `1 - C(N-M,t)/C(N,t)`, interpreted as 1 if t>N-M. The full permutation's CPU generation cost is included in wall time, although only inspected candidates count as evaluations.
- **Exhaustive-first search:** lexicographic integers 0..N-1, stopping at the first success; if none exists it examines all N. Its ordering can strongly favor some fixtures. The setup audit always scans all N to count M, even if a feasible schedule appears early.

The quantum iteration plan and a common **decision-query cap** are fixed *before observing any outputs*: cap = planned oracle calls + maximum output-verification checks. The random baseline receives that same cap, limited by N distinct candidates. Both searches can stop early, so actual query counts can differ. Exhaustive-first is a separate full-space-cap comparator. With one attempt and known M, the random baseline cap is k+1. This comparison includes verification and exposes the query/cost tradeoff; it does not equate a classical predicate evaluation with the gate cost of a phase oracle.

| Output metric | Meaning and limits |
|---|---|
| `grover_iterations`, `logical_oracle_calls` | Total actual iterations/complete phase-oracle invocations; one oracle per iteration, summed over attempted circuits |
| `candidate_evaluations` | Number of classical calls to the complete feasibility checker; for Grover these are output-verification checks, not superposition components |
| `constraint_evaluations`, `..._by_rule` | Actual classical predicate calls with short circuiting; quantum rows count only classical verification, never all branches of a superposition |
| `reversible_rule_blocks` | Structural reversible rule-block uses: **4m per complete oracle**, including flag uncomputation and reverse counting; not measured Python predicate calls |
| `total_decision_queries` | Logical oracle calls + classical candidate checks; useful query accounting, not a gate/time equivalence |
| `success_rate` | Fraction of repeated trials with a classically verified feasible sample, under the stated attempt/query cap; finite-trial uncertainty remains |
| `simulator_seconds` | Aer job submission, execution and result retrieval, summed per run; excludes building and transpiling |
| `transpile_seconds` | Actual newly incurred compilation time; cached circuits cost zero on later uses |
| `search_wall_seconds` | Timed algorithm/search work, including verification and any newly incurred transpilation; excludes one-time oracle construction and audit |
| `ancillas`, `circuit_qubits` | Work qubits and total allocated qubits; do not infer zero work merely from data-qubit count |
| `executed_cx`, `executed_gates` | Sum of transpiled circuit CX counts / instruction counts over attempts; gates include measurements in the instruction total |
| `max_circuit_depth` | Maximum transpiled circuit depth among actual attempts, not hardware runtime |
| `synthesis_predicate_evaluations` | One-time sum of `2^support_size` across active rules, outside candidate search |
| `synthesis_minterms` | Chosen minterm patterns before full-oracle repetition and basis decomposition |

`setup.csv` holds audit and construction costs, including abstract oracle size/depth (local predicate gates still count as single composite instructions there). `trials.csv` holds detailed observations and sizes after decomposition. `summary.csv` aggregates successes and means. `environment.json` records versions and invocation settings. Summary simulator times do not include classical baseline times; use search wall time for those. Package imports, initial backend creation, and CSV writing are outside the search timers. Single-thread simulation reduces timing variation but does not make timings hardware-independent.

A failed Grover row always says **`no_verified_sample`**, including for an audited UNSAT fixture. Only exhaustive evidence says `proven_unsat`; the audit's conclusion is carried separately. For unknown real instances outside an exhaustive budget, failure remains inconclusive.

## What these results can establish

For sparse solutions, Grover can need O(sqrt(N/M)) logical oracle calls versus O(N/M) expected classical unstructured candidate checks. One quantum oracle can contain many gates, and different rules with the same query count can have radically different cost. Counter arithmetic, reversible predicate work, multi-controlled-gate decomposition, construction, transpilation, and output verification can erase practical query savings. Inspect M, local support sizes, CX/depth, ancillas, compile costs and search times together.

Statevector simulation is a classical calculation with state size exponential in **total** circuit qubits. Its wall time is **not evidence of a quantum speedup** or an estimate of hardware wall time. This implementation's table-based preprocessing can itself equal or exceed exhaustive classical search. It does not compare against specialized scheduling solvers, pruning, SAT/CP-SAT/MILP or business heuristics; these may exploit structure far better than unstructured search.

Reference APIs: [Qiskit QuantumCircuit](https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.circuit.QuantumCircuit), [Grover operator and phase-oracle convention](https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.circuit.library.grover_operator), [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).
