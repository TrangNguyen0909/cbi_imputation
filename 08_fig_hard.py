import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]; S = pd.read_csv(OUT + '/results_hard_sim_summary.csv')
BLUE, ORANGE, GRAY, AQUA, TXT, TXT2, GRID = '#2a78d6', '#eb6834', '#9b9a95', '#1baf7a', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'xtick.color': TXT2, 'ytick.color': TXT2, 'axes.spines.top': False, 'axes.spines.right': False})
M = [('equal_split_remainder', 'National total only (equal split of remainder)', GRAY), ('ML_only', 'ML only', BLUE), ('ML+reconcile_bound', 'ML + reconciliation (proposed)', ORANGE)]
SC = [('A_company', 'A. Company-level hiding\n(narrow national ranges)'), ('B_empirical_fraction', 'B. Realistic per-chemical hiding\n(narrow national ranges)'), ('C_coarse_range', 'C. As B, ranges coarsened\nto a 10× bracket')]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
for k, (sub, title) in enumerate([('chem_partly_hidden', 'Chemicals with some public sites (74% of real CBI rows)'), ('chem_fully_hidden', 'Chemicals with no public site (26% of real CBI rows)')]):
    a = ax[k]
    for j, (sc, _) in enumerate(SC):
        for o, (mt, lab, c) in enumerate(M):
            r = S[(S.scenario == sc) & (S.subset == sub) & (S.method == mt)].iloc[0]
            v, e = r.MAE_mean, r.MAE_std; yy = j + (o - 1) * 0.26
            a.barh(yy, min(v, 1.2), height=0.24, color=c, label=lab if (k == 0 and j == 0) else None)
            a.text(min(v, 1.2) + 0.02, yy, f'{v:.2f}' + (' →' if v > 1.2 else ''), va='center', fontsize=8, color=TXT)
    a.set_yticks(range(3)); a.set_yticklabels([s[1] for s in SC], color=TXT); a.invert_yaxis(); a.set_xlim(0, 1.4)
    a.set_title(title, loc='left', fontsize=10, color=TXT); a.xaxis.grid(True, color=GRID); a.set_axisbelow(True); a.set_xlabel('Row-level MAE, log10 lb (lower is better)', color=TXT2)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle('Fig. 3. Harder pseudo-CBI simulations (5 repetitions each): ML and the national constraint are complementary', x=0.01, ha='left', fontsize=11, color=TXT)
plt.tight_layout(rect=(0, 0.07, 1, 0.94)); plt.savefig(OUT + '/fig3_hard_simulations.png', dpi=200, bbox_inches='tight')
