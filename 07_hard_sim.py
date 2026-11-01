"""Harder pseudo-CBI simulations for the reconciliation step.
A: company-level hiding (20% of parents, size-weighted)            [previous scenario]
B: per-chemical hidden fraction drawn from the REAL distribution of CBI fractions (incl. 100%),
   hidden sites chosen preferentially from large companies (keeps the MNAR pattern)
C: as B, but the national range is coarsened to an order-of-magnitude bracket [10^k, 10^(k+1))
Competitors: ML only | ML + reconcile to nearest bound | ML + reconcile to remainder midpoint | equal split of remainder (no ML)."""
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from scipy.stats import spearmanr
exec(open('03_mask_aug.py').read().split("D = X[X.status == 'public']")[0])
D = X[X.status == 'public'].reset_index(drop=True); y = D.y.values
allpub = X.groupby('CHEMICAL ID').status.apply(lambda s: (s == 'public').all())
ok = D['CHEMICAL ID'].isin(set(allpub[allpub].index)).values
# empirical per-chemical CBI fraction among chemicals that have CBI rows
fr = X.groupby('CHEMICAL ID').status.apply(lambda s: (s == 'CBI').mean()); fr = fr[fr > 0].values
nsite = D.groupby('CHEMICAL ID').size(); psz = D.parent.astype(str).map(D.parent.astype(str).value_counts()).values.astype(float)
nlo_t = D.groupby('CHEMICAL ID').nat_lo.first(); nhi_t = D.groupby('CHEMICAL ID').nat_hi.first()
def reconcile(T, known, lo, hi, mode):
    lin = 10 ** T.pred; tot = lin.groupby(T['CHEMICAL ID']).transform('sum')
    k = T['CHEMICAL ID'].map(known).fillna(0)
    rlo = (T['CHEMICAL ID'].map(lo) - k).clip(lower=0); rhi = (T['CHEMICAL ID'].map(hi) - k).clip(lower=1)
    if mode == 'bound': tgt = np.where(tot < rlo, rlo, np.where(tot > rhi, rhi, tot))
    else: tgt = np.sqrt(np.maximum(rlo, 1) * rhi)  # always rescale to geometric midpoint of remainder
    return np.log10(np.clip(lin * tgt / tot, 1, None))
def equal_split(T, known, lo, hi):
    k = T['CHEMICAL ID'].map(known).fillna(0); nh = T.groupby('CHEMICAL ID').y.transform('size')
    rlo = (T['CHEMICAL ID'].map(lo) - k).clip(lower=0); rhi = (T['CHEMICAL ID'].map(hi) - k).clip(lower=1)
    return np.log10(np.clip(np.sqrt(np.maximum(rlo, 1) * rhi) / nh, 1, None))
def score(T, col):
    e = T[col] - T.y; ct = T.groupby('CHEMICAL ID').apply(lambda g: abs(np.log10((10 ** g[col]).sum()) - np.log10(g.pv.sum())))
    return dict(MAE=float(abs(e).mean()), bias=float(e.mean()), within1=float(np.mean(abs(b(T[col]) - b(T.y)) <= 1)),
                spearman_within_chem=float(np.nanmean([spearmanr(g.y, g[col])[0] for _, g in T.groupby('CHEMICAL ID') if len(g) >= 3 and g[col].nunique() > 1])),
                chem_total_MAE=float(ct.mean()))
rows = []
for r in range(5):
    rs = np.random.default_rng(500 + r)
    # --- scenario A
    sz = D.groupby(D.parent.astype(str)).size(); pr = (sz ** 0.5) / (sz ** 0.5).sum()
    pick = set(rs.choice(sz.index, int(0.2 * len(sz)), replace=False, p=pr.values))
    hidA = D.parent.astype(str).isin(pick).values & ok
    # --- scenario B: per chemical draw a hidden fraction, choose sites with prob ∝ sqrt(company size)
    hidB = np.zeros(len(D), bool)
    for cid, idx in D[ok].groupby('CHEMICAL ID').indices.items():
        idx = D[ok].index[idx] if False else np.flatnonzero(ok)[idx]
        if rs.random() > 0.6: continue                       # ~40% of chemicals get some hidden sites
        k = max(1, int(round(rs.choice(fr) * len(idx))))
        w = np.sqrt(psz[idx]); hidB[rs.choice(idx, size=min(k, len(idx)), replace=False, p=w / w.sum())] = True
    for scen, hid in [('A_company', hidA), ('B_empirical_fraction', hidB)]:
        m = lgb.LGBMRegressor(**P).fit(prep(augment(D[~hid]))[FEAT], np.tile(y[~hid], 3))
        T = D[hid].copy(); T['pred'] = m.predict(prep(apply_mask(T, sample_masks(len(T))))[FEAT])
        # national-level features of hidden chemicals stay public; n_sites etc. are unchanged (public in real data)
        known = D[~hid].groupby('CHEMICAL ID').pv.sum()
        ranges = {'': (nlo_t, nhi_t)}
        if scen.startswith('B'):
            tt = D.groupby('CHEMICAL ID').pv.sum(); kexp = np.floor(np.log10(tt.clip(lower=1)))
            ranges['C_coarse_range'] = (10 ** kexp, 10 ** (kexp + 1))
        for tag, (lo, hi) in ranges.items():
            name = tag or scen
            T['p_bound'] = reconcile(T, known, lo, hi, 'bound'); T['p_mid'] = reconcile(T, known, lo, hi, 'mid'); T['p_eq'] = equal_split(T, known, lo, hi)
            fully = T['CHEMICAL ID'].map(known).isna()
            for subset, S in [('all', T), ('chem_fully_hidden', T[fully]), ('chem_partly_hidden', T[~fully])]:
                if len(S) < 30: continue
                for meth, col in [('ML_only', 'pred'), ('ML+reconcile_bound', 'p_bound'), ('ML+reconcile_mid', 'p_mid'), ('equal_split_remainder', 'p_eq')]:
                    rows.append(dict(rep=r, scenario=name, subset=subset, method=meth, n=len(S), **score(S, col)))
R = pd.DataFrame(rows); R.to_csv(OUT + '/results_hard_sim_raw.csv', index=False)
S = R.groupby(['scenario', 'subset', 'method']).agg(['mean', 'std']).drop(columns='rep')
S.columns = [f'{a}_{b}' for a, b in S.columns]; S.to_csv(OUT + '/results_hard_sim_summary.csv')
pd.set_option('display.width', 250)
print(S[[c for c in S.columns if c.endswith('_mean')]].round(3).to_string())
