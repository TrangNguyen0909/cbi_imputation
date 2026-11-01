import json, pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]
r2, r3, r4, r5 = [json.load(open(f'{OUT}/results_{k}.json')) for k in ['model', 'mask_aug', 'company_sim', 'apply']]
BLUE, ORANGE, AQUA, GRAY, TXT, TXT2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#9b9a95', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'xtick.color': TXT2, 'ytick.color': TXT2,
                     'axes.spines.top': False, 'axes.spines.right': False})
# Fig 1: evaluation ladder
rows = [('Unseen chemical\n(clean test)', r2['unseen_chemical']['equal_split'], r2['unseen_chemical']['lightgbm'], None),
        ('Unseen chemical\n(CBI-masked test)', r3['simulated_CBI_test_unseen_chemical']['equal_split'], r3['simulated_CBI_test_unseen_chemical']['augmented_model'], None),
        ('Company-level\npseudo-CBI', None, {'MAE': r4['company_pseudoCBI_sim_mean']['row_MAE_ml'], 'within1': r4['company_pseudoCBI_sim_mean']['row_within1_ml']},
         {'MAE': r4['company_pseudoCBI_sim_mean']['row_MAE_reconciled'], 'within1': r4['company_pseudoCBI_sim_mean']['row_within1_reconciled']})]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k, (met, title) in enumerate([('within1', 'Share within ±1 volume bin (higher is better)'), ('MAE', 'Mean absolute error, log10 lb (lower is better)')]):
    a = ax[k]
    for j, (lab, eq, ml, rec) in enumerate(rows):
        pts = [('Equal-split baseline', eq, GRAY), ('ML (LightGBM, mask-augmented)', ml, BLUE), ('ML + national reconciliation', rec, ORANGE)]
        for o, (nm, d, c) in enumerate(pts):
            if d is None or d.get(met) is None or (isinstance(d.get(met), float) and np.isnan(d.get(met))): continue
            a.barh(j + (o - 1) * 0.26, d[met], height=0.24, color=c, label=nm if (k == 0 and j == 2) or (k == 0 and j == 0 and o == 0) else None)
            a.text(d[met] * 1.01, j + (o - 1) * 0.26, f'{d[met]:.2f}', va='center', fontsize=8, color=TXT)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows] if k == 0 else [''] * 3, color=TXT); a.invert_yaxis()
    a.set_title(title, loc='left', fontsize=10, color=TXT); a.xaxis.grid(True, color=GRID); a.set_axisbelow(True)
    a.set_xlim(0, (1.08 if met == 'within1' else max(1.15, a.get_xlim()[1] * 1.1)))
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('Fig. 1. Imputing confidential site-level production volumes: harder test settings, and the gain from reconciliation', x=0.01, ha='left', fontsize=11, color=TXT)
plt.tight_layout(rect=(0, 0.07, 1, 0.94)); plt.savefig(OUT + '/fig1_evaluation.png', dpi=200, bbox_inches='tight'); plt.close()
# Fig 2: hidden share TSCA high-priority
hp = pd.read_csv(OUT + '/tsca_high_priority_summary.csv').sort_values('hidden_share')
fig, a = plt.subplots(figsize=(8, 5.4))
a.barh(range(len(hp)), hp.hidden_share * 100, color=BLUE, height=0.6)
for j, (_, r) in enumerate(hp.iterrows()):
    a.text(r.hidden_share * 100 + 1, j, f"{r.hidden_share*100:.0f}%  ({int(r.n_sites_cbi)} of {int(r.n_sites_cbi + r.n_sites_public)} sites)", va='center', fontsize=8, color=TXT2)
fig.text(0.01, -0.03, f"For comparison, all PFAS combined: {r5['PFAS']['hidden_share_volume']*100:.0f}% of volume at CBI sites. Values at 100% have no public site; there the share follows from the national total alone.", fontsize=8, color=TXT2)
a.set_yticks(range(len(hp))); a.set_yticklabels(hp.name, color=TXT); a.set_xlim(0, 125); a.set_xticks([0, 25, 50, 75, 100])
a.set_xlabel('Share of 2023 national volume at confidential sites (imputed, %)', color=TXT2); a.xaxis.grid(True, color=GRID); a.set_axisbelow(True)
a.set_title('Fig. 2. How much of each TSCA high-priority chemical is produced at CBI sites', loc='left', fontsize=11, color=TXT)
plt.tight_layout(); plt.savefig(OUT + '/fig2_hidden_share_priority.png', dpi=200, bbox_inches='tight')
