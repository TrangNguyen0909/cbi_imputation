"""Impute real CBI rows: augmented LightGBM + CQR intervals + national-aggregate reconciliation; summaries for PFAS & TSCA high-priority chemicals."""
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
exec(open('03_mask_aug.py').read().split("D = X[X.status == 'public']")[0])
PF = sys.argv[2] if len(sys.argv) > 2 else None
D = X[X.status == 'public'].reset_index(drop=True); y = D.y.values
C = X[X.status == 'CBI'].reset_index(drop=True)
Tr = prep(augment(D)); yt = np.tile(y, 3)
mk = lambda **kw: lgb.LGBMRegressor(**{**P, **kw}).fit(Tr[FEAT], yt)
Cp = prep(C)
C['pred'] = mk().predict(Cp[FEAT])
q05 = mk(objective='quantile', alpha=0.05).predict(Cp[FEAT]); q95 = mk(objective='quantile', alpha=0.95).predict(Cp[FEAT])
O = pd.read_parquet(OUT + '/public_oof_masked.parquet')
s = np.maximum(O.oof_q05 - O.y, O.y - O.oof_q95); qh = np.quantile(s, np.ceil((len(s) + 1) * 0.9) / len(s))
C['lo90'], C['hi90'] = np.minimum(q05 - qh, C.pred), np.maximum(q95 + qh, C.pred)
# reconciliation with national range minus public sum
known = D.groupby('CHEMICAL ID').pv.sum()
g = C.groupby('CHEMICAL ID'); tot = g.pred.transform(lambda v: (10 ** v).sum())
nlo, nhi = C['CHEMICAL ID'].map(X.groupby('CHEMICAL ID').nat_lo.first()), C['CHEMICAL ID'].map(X.groupby('CHEMICAL ID').nat_hi.first())
kn = C['CHEMICAL ID'].map(known).fillna(0)
rlo, rhi = (nlo - kn).clip(lower=0), (nhi - kn).clip(lower=1)
f = np.where(nhi.isna(), 1.0, np.where(tot < rlo, rlo / tot, np.where(tot > rhi, rhi / tot, 1.0)))
sh = np.log10(f)
C['pred_rec'] = C.pred + sh; C['lo90_rec'] = C.lo90 + sh; C['hi90_rec'] = C.hi90 + sh; C['rec_factor'] = f
C['pv_hat_lb'] = 10 ** C.pred_rec
names = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else 'chem_names.csv') if False else None
C.to_csv(OUT + '/imputed_cbi_rows.csv', index=False)
res = dict(n_cbi_rows=len(C), conformal_margin_log10=float(qh), mean_interval_width_log10=float((C.hi90 - C.lo90).mean()),
           frac_rows_reconciled=float(np.mean(f != 1)), median_reconcile_factor_up=float(np.median(f[f > 1])) if (f > 1).any() else 1.0,
           frac_reconciled_upward=float(np.mean(f > 1)), frac_reconciled_downward=float(np.mean(f < 1)),
           mean_shift_log10=float(np.mean(sh)))
# hidden share of national volume
nat = X.groupby('CHEMICAL ID').agg(lo=('nat_lo', 'first'), hi=('nat_hi', 'first')); nat['pub'] = known.reindex(nat.index).fillna(0)
nat['imp'] = C.groupby('CHEMICAL ID').pv_hat_lb.sum().reindex(nat.index).fillna(0)
nat['hidden_share'] = nat.imp / (nat.imp + nat.pub)
res['volume_weighted_hidden_share_all'] = float(nat.imp.sum() / (nat.imp.sum() + nat.pub.sum()))
res['chemicals_with_any_CBI_row'] = int((nat.imp > 0).sum())
res['chemicals_majority_hidden'] = int((nat.hidden_share > 0.5).sum())
if PF:
    pf = pd.read_csv(PF); ids = set(pf.loc[pf.group.str.startswith('PFAS', na=False), 'id'].astype(str))
    sub = nat[nat.index.astype(str).isin(ids)]
    res['PFAS'] = dict(n_chem=len(sub), hidden_share_volume=float(sub.imp.sum() / max(sub.imp.sum() + sub.pub.sum(), 1)),
                       imputed_Mlb=float(sub.imp.sum() / 1e6), public_Mlb=float(sub.pub.sum() / 1e6))
HP = {'106-46-7':'p-Dichlorobenzene','107-06-2':'1,2-Dichloroethane','156-60-5':'trans-1,2-Dichloroethylene','95-50-1':'o-Dichlorobenzene',
      '79-00-5':'1,1,2-Trichloroethane','78-87-5':'1,2-Dichloropropane','75-34-3':'1,1-Dichloroethane','106-99-0':'1,3-Butadiene',
      '50-00-0':'Formaldehyde','85-44-9':'Phthalic anhydride','79-94-7':'TBBPA','115-96-8':'TCEP','13674-87-8':'TDCPP','1222-05-5':'HHCB',
      '84-74-2':'DBP','85-68-7':'BBP','117-81-7':'DEHP','84-69-5':'DIBP','84-61-7':'DCHP','115-86-6':'Triphenyl phosphate'}
hp = nat.reindex([k for k in HP if k in nat.index]).copy(); hp['name'] = [HP[k] for k in hp.index]
hp['n_sites_public'] = D.groupby('CHEMICAL ID').size().reindex(hp.index).fillna(0).astype(int)
hp['n_sites_cbi'] = C.groupby('CHEMICAL ID').size().reindex(hp.index).fillna(0).astype(int)
hp = hp[['name', 'n_sites_public', 'n_sites_cbi', 'lo', 'hi', 'pub', 'imp', 'hidden_share']]
hp.to_csv(OUT + '/tsca_high_priority_summary.csv'); nat.to_csv(OUT + '/chemical_level_summary.csv')
json.dump(res, open(OUT + '/results_apply.json', 'w'), indent=1)
print(json.dumps(res, indent=1)); print(hp.assign(lo=hp.lo/1e6, hi=hp.hi/1e6, pub=hp.pub/1e6, imp=hp.imp/1e6).round(2).to_string())
