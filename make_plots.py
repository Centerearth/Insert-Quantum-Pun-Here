"""Turn results/*.csv into slide-ready PNGs in results/plots/.   python make_plots.py"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

R = Path('results'); P = R / 'plots'; P.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'figure.facecolor': 'white', 'savefig.dpi': 200,
                     'lines.linewidth': 2})


def rows(name):
    path = R / name
    return list(csv.DictReader(open(path))) if path.exists() else []


def num(r, k):
    return float(r[k])


# 1. Iteration sweep: simulation vs theory, one panel per size
sweep = rows('iteration_sweep.csv')
if sweep:
    sizes = sorted({int(r['data_qubits']) for r in sweep})
    fig, axes = plt.subplots(1, len(sizes), figsize=(13, 4), sharey=True)
    for ax, n in zip(axes, sizes):
        d = [r for r in sweep if int(r['data_qubits']) == n]
        k = [int(r['iterations']) for r in d]
        th = np.arcsin(np.sqrt(num(d[0], 'M') / num(d[0], 'N')))
        kk = np.linspace(0, max(k), 300)
        ax.plot(kk, np.sin((2 * kk + 1) * th) ** 2, color=BLUE, alpha=0.45, label='theory sin²((2k+1)θ)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(k, [num(r, 'exact_success') for r in d], 'o', color=BLUE, mfc='white', ms=8, mew=2, label='simulated')
        kb = int(d[0]['optimal_iterations'])
        ax.axvline(kb, color=INK2, ls=':', lw=1.2)
        ax.text(kb + 0.15, 0.04, f'best k = {kb}', color=INK2, fontsize=10)
        ax.set_title(f'{n} qubits · {int(d[0]["M"])} valid of {int(d[0]["N"])}', color=INK, fontsize=12)
        ax.set_xlabel('Grover iterations k')
    axes[0].set_ylabel('P(valid schedule)'); axes[0].set_ylim(0, 1.05)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='upper right', ncol=2, fontsize=10)
    fig.suptitle('Too many iterations overshoots: success rises, peaks, then falls', color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(P / '1_iteration_sweep.png'); plt.close(fig)

# 2. Queries to find a valid schedule vs search-space size
comp = rows('classical_comparison.csv')
if comp:
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [('brute force (random order)', ORANGE, 'Brute force'),
              ('Grover (repeat until valid)', BLUE, 'Grover'),
              ('backtracking', AQUA, 'Backtracking')]
    for method, color, label in series:
        d = sorted((r for r in comp if r['method'] == method), key=lambda r: int(r['N']))
        x = [int(r['N']) for r in d]; y = [num(r, 'mean') for r in d]
        ax.plot(x, y, '-o', color=color, ms=8, label=label)
        ax.text(x[-1] * 1.12, y[-1], f'{label}  {y[-1]:.0f}', color=INK, va='center', fontsize=11)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlim(12, 700); ax.set_xlabel('possible schedules N (log scale)')
    ax.set_ylabel('candidates checked / oracle calls (log scale)')
    ax.text(16, 1.3, 'CP-SAT: solved in presolve, 0 search branches, ~1 ms at every size', color=INK2, fontsize=10)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('Grover beats brute force (√N vs N), but a real solver needs no search at all',
                 color=INK, loc='left', fontsize=12)
    fig.tight_layout(); fig.savefig(P / '2_queries_vs_size.png'); plt.close(fig)

# 3. Constraint ladder: two measures -> two panels (no dual axis)
lad = rows('constraint_ladder.csv')
if lad:
    m = [int(r['rules']) for r in lad]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.bar(m, [int(r['cx_per_iteration']) for r in lad], color=BLUE, width=0.7)
    a1.set_xlabel('business rules in the oracle (8 qubits)'); a1.set_ylabel('CX gates per Grover iteration')
    a1.set_title('Each rule makes every iteration more expensive', color=INK, loc='left', fontsize=12)
    a1.annotate('counter grows\n3 → 4 qubits', xy=(8, int(lad[7]['cx_per_iteration'])),
                xytext=(4.2, 1900), color=INK2, fontsize=10, arrowprops=dict(arrowstyle='->', color=INK2))
    a1.annotate('$85 budget rule\n(weighted sum)', xy=(11, int(lad[10]['cx_per_iteration'])),
                xytext=(8.6, 2550), color=INK2, fontsize=10, arrowprops=dict(arrowstyle='->', color=INK2))
    a1.set_ylim(0, 2800)
    a2.bar(m, [int(r['cx_total']) for r in lad], color=ORANGE, width=0.7)
    for x, r in zip(m, lad):
        a2.text(x, int(r['cx_total']) + 250, f'k={r["iterations"]}', ha='center', fontsize=9, color=INK2)
    a2.set_xlabel('business rules in the oracle (8 qubits)'); a2.set_ylabel('total CX gates (all iterations)')
    a2.set_title('…and fewer valid schedules means more iterations', color=INK, loc='left', fontsize=12)
    for a in (a1, a2):
        a.set_xticks(m); a.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(P / '3_constraint_ladder.png'); plt.close(fig)

# 4. Noise: one panel per size
noise = rows('noise.csv')
if noise:
    sizes = sorted({int(r['data_qubits']) for r in noise})
    fig, axes = plt.subplots(1, len(sizes), figsize=(13, 4.3), sharey=True)
    axes = axes if len(sizes) > 1 else [axes]
    for ax, n in zip(axes, sizes):
        d = sorted((r for r in noise if int(r['data_qubits']) == n and num(r, 'two_qubit_error_rate') > 0),
                   key=lambda r: num(r, 'two_qubit_error_rate'))
        x = [num(r, 'two_qubit_error_rate') for r in d]
        ax.errorbar(x, [num(r, 'measured_success') for r in d], yerr=[2 * num(r, 'std_error') for r in d],
                    fmt='o', color=BLUE, ms=8, mfc='white', mew=2, capsize=3, label='noisy simulation')
        ax.plot(x, [num(r, 'model_success') for r in d], color=BLUE, alpha=0.5, label='simple error model')
        ideal, rnd = num(d[0], 'ideal_success'), num(d[0], 'random_guess_success')
        ax.axhline(ideal, color=AQUA, ls='--', lw=1.5); ax.text(3e-3, ideal - 0.07, 'noiseless', color=INK2, fontsize=10, ha='left')
        ax.axhline(rnd, color=ORANGE, ls='--', lw=1.5); ax.text(x[0], rnd + 0.03, 'random guess', color=INK2, fontsize=10, ha='right')
        ax.axvspan(1e-3 * 0.7, 3e-3, color=GRID, alpha=0.6, lw=0)
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_title(f'{n} qubits · {d[0]["cx_total"]} CX gates', color=INK, fontsize=12)
        ax.set_xlabel('two-qubit gate error rate')
    axes[0].text(2.6e-3, 0.62, "today's\nbest\nhardware", color=INK2, fontsize=9)
    axes[0].set_ylabel('P(valid schedule)'); axes[0].set_ylim(0, 1.08)
    axes[-1].legend(frameon=False, loc='center left', fontsize=10)
    fig.suptitle('At realistic error rates the answer drowns in noise; it takes ~10⁻⁵ errors to keep it',
                 color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(P / '4_noise.png'); plt.close(fig)

# 5. Real size: CP-SAT solve time
real = rows('real_size_cpsat.csv')
if real:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f'{r["staff"]} staff\n2^{r["binary_variables"]} schedules' for r in real]
    t = [num(r, 'wall_seconds') * 1000 for r in real]
    ax.bar(range(len(real)), t, color=AQUA, width=0.6)
    for i, v in enumerate(t):
        ax.text(i, v + max(t) * 0.02, f'{v:.0f} ms', ha='center', color=INK)
    ax.set_xticks(range(len(real))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('CP-SAT time to a valid week (ms)'); ax.grid(axis='x', visible=False)
    ax.set_title('Real facility weeks (7 days × 3 shifts) solve in milliseconds on a laptop core',
                 color=INK, loc='left', fontsize=12)
    fig.tight_layout(); fig.savefig(P / '5_real_size_cpsat.png'); plt.close(fig)

print('plots written to', P.resolve())
