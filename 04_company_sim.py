"""(a) leave-company-out evaluation; (b) company-level pseudo-CBI simulation with national-aggregate reconciliation."""
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
exec(open('03_mask_aug.py').read().split("D = X[X.status == 'public']")[0])
D = X[X.status == 'public'].reset_index(drop=True); y = D.y.values
res = {}
def fit(df, yy, **kw): return lgb.LGBMRegressor(**{**P, **kw}).fit(prep(df)[FEAT], yy)
# (a) leave-company-out
pred = np.zeros(len(D))
for tr, te in GroupKFold(5).split(D, y, D.parent.astype(str)):
    m = fit(augment(D.loc[tr]), np.tile(y[tr], 3)); pred[te] = m.predict(prep(apply_mask(D.loc[te], sample_masks(len(te))))[FEAT])
res['leave_company_out'] = {**metrics(y, pred), 'bias_log10': float(np.mean(pred - y))}
big = D.n_rows_parent >= D.n_rows_parent.quantile(0.9)
res['leave_company_out_large_companies'] = {**metrics(y[big], pred[big]), 'bias_log10': float(np.mean(pred[big] - y[big]))}
# (b) pseudo-CBI simulation on chemicals with no real CBI rows (national range == sum of public sites)
allpub = X.groupby('CHEMICAL ID').status.apply(lambda s: (s == 'public').all())
chem_ok = set(allpub[allpub].index)
def reconcile(df, pcol, known, lo, hi):
    """Scale pseudo-CBI predictions (linear space) of each chemical so that known + sum(pred) lies in [lo, hi]."""
    df = df.copy(); df['lin'] = 10 ** df[pcol]
    tot = df.groupby('CHEMICAL ID').lin.transform('sum')
    rlo = (df['CHEMICAL ID'].map(lo) - df['CHEMICAL ID'].map(known)).clip(lower=0)
    rhi = (df['CHEMICAL ID'].map(hi) - df['CHEMICAL ID'].map(known)).clip(lower=1)
    f = np.where(tot < rlo, rlo / tot, np.where(tot > rhi, rhi / tot, 1.0))
    return np.log10(np.clip(df.lin * f, 1, None)), float(np.mean(f != 1))
sims = []
parents = D.parent.astype(str).unique()
for r in range(5):
    rs = np.random.default_rng(100 + r)
    # company-level policy: pick parents at random, probability rising with company size (mimics observed pattern)
    sz = D.groupby(D.parent.astype(str)).size(); pr = (sz ** 0.5) / (sz ** 0.5).sum()
    pick = set(rs.choice(sz.index, size=int(0.2 * len(sz)), replace=False, p=pr.values))
    pseudo = D.parent.astype(str).isin(pick).values & D['CHEMICAL ID'].isin(chem_ok).values
    tr = ~D.parent.astype(str).isin(pick).values
    m = fit(augment(D[tr]), np.tile(y[tr], 3))
    T = D[pseudo].copy(); T['pred'] = m.predict(prep(apply_mask(T, sample_masks(len(T))))[FEAT])
    known = D[~pseudo].groupby('CHEMICAL ID').pv.sum()
    nlo = D.groupby('CHEMICAL ID').nat_lo.first(); nhi = D.groupby('CHEMICAL ID').nat_hi.first()
    T['pred_rec'], frac_adj = reconcile(T, 'pred', known.reindex(nlo.index).fillna(0), nlo, nhi)
    # chemical-level totals error (log10 of hidden total)
    ct = T.groupby('CHEMICAL ID').agg(true=('pv', 'sum'), p=('pred', lambda v: (10 ** v).sum()), pr=('pred_rec', lambda v: (10 ** v).sum()))
    sims.append(dict(n_pseudo_rows=int(pseudo.sum()), frac_chem_adjusted=frac_adj,
        row_MAE_ml=float(np.mean(abs(T.pred - T.y))), row_MAE_reconciled=float(np.mean(abs(T.pred_rec - T.y))),
        row_bias_ml=float(np.mean(T.pred - T.y)), row_bias_reconciled=float(np.mean(T.pred_rec - T.y)),
        row_within1_ml=float(np.mean(abs(b(T.pred) - b(T.y)) <= 1)), row_within1_reconciled=float(np.mean(abs(b(T.pred_rec) - b(T.y)) <= 1)),
        chem_total_MAE_ml=float(np.mean(abs(np.log10(ct.p) - np.log10(ct.true)))), chem_total_MAE_reconciled=float(np.mean(abs(np.log10(ct.pr) - np.log10(ct.true))))))
S = pd.DataFrame(sims); res['company_pseudoCBI_sim_mean'] = S.mean().to_dict(); res['company_pseudoCBI_sim_sd'] = S.std().to_dict()
json.dump(res, open(OUT + '/results_company_sim.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
