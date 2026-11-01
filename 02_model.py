"""Covariate-shift diagnosis, baselines, LightGBM, cross-validation."""
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT = sys.argv[1]
X = pd.read_parquet(OUT + '/site_table.parquet')
CAT = ['activity','state','workers','maxconc','byp','recycled','never_at_site','foreign_cc','phys_form','naics6','naics3','id_type',
       'ind_fc','ind_sect','ind_type','ind_sites','ind_workers','con_cat','con_use','kids']
NUM = ['n_cbi_fields','log_onsite','onsite_cbi','log_exported','exported_cbi','nat_log_lo','nat_log_hi','nat_log_per_site','n_sites_chem',
       'n_chem_site','n_rows_parent','frac_cbi_chem_others','n_ind','n_ind_sect','n_con']
FEAT = CAT + NUM
for c in CAT: X[c] = X[c].astype(object).fillna('NA').astype(str).astype('category')
X = X[X.status.isin(['public','CBI'])].reset_index(drop=True)
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, cat_smooth=20, verbose=-1)
res = {}
# 1) adversarial validation: CBI vs public
isc = (X.status == 'CBI').astype(int).values
oof = np.zeros(len(X)); imp = 0
FEAT_ADV = [f for f in FEAT if f not in ('n_cbi_fields','frac_cbi_chem_others')]  # avoid trivially encoding CBI status
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, isc):
    mdl = lgb.LGBMClassifier(**{**P, 'n_estimators': 300}).fit(X.loc[tr, FEAT_ADV], isc[tr])
    oof[te] = mdl.predict_proba(X.loc[te, FEAT_ADV])[:, 1]; imp = imp + mdl.booster_.feature_importance('gain')
res['adversarial_auc'] = roc_auc_score(isc, oof)
res['adversarial_top_features'] = pd.Series(imp, FEAT_ADV).sort_values(ascending=False).head(8).round(0).to_dict()
X['p_cbi'] = oof
# 2) regression on public rows
D = X[X.status == 'public'].reset_index(drop=True); y = D.y.values
BINS = [0, np.log10(25e3), 5, 6, 7, 8, 99]
b = lambda v: np.digitize(v, BINS[1:-1])
def metrics(yt, yp):
    return dict(MAE=float(np.mean(abs(yt - yp))), RMSE=float(np.sqrt(np.mean((yt - yp) ** 2))), bin_acc=float(np.mean(b(yt) == b(yp))),
                within1=float(np.mean(abs(b(yt) - b(yp)) <= 1)), spearman=float(spearmanr(yt, yp)[0]))
schemes = {'unseen_chemical': GroupKFold(5).split(D, y, D['CHEMICAL ID']), 'random_rows': KFold(5, shuffle=True, random_state=0).split(D)}
preds = {}
for sname, split in schemes.items():
    pr = {'global_median': np.zeros(len(D)), 'equal_split': D.nat_log_per_site.values, 'lightgbm': np.zeros(len(D))}
    for tr, te in split:
        pr['global_median'][te] = np.median(y[tr])
        mdl = lgb.LGBMRegressor(**P).fit(D.loc[tr, FEAT], y[tr]); pr['lightgbm'][te] = mdl.predict(D.loc[te, FEAT])
    res[sname] = {k: metrics(y, v) for k, v in pr.items()}
    # performance on CBI-like public rows (top 20% p_cbi)
    hi = D.p_cbi >= D.p_cbi.quantile(0.8)
    res[sname + '_CBIlike_top20pct'] = {k: metrics(y[hi], v[hi]) for k, v in pr.items()}
    preds[sname] = pr['lightgbm']
D['pred_unseen'] = preds['unseen_chemical']; D['pred_random'] = preds['random_rows']
D.to_parquet(OUT + '/public_oof.parquet', index=False); X[['row_id', 'p_cbi']].to_parquet(OUT + '/p_cbi.parquet', index=False)
full = lgb.LGBMRegressor(**P).fit(D[FEAT], y)
res['final_importance_gain'] = pd.Series(full.booster_.feature_importance('gain'), FEAT).sort_values(ascending=False).head(12).round(0).to_dict()
json.dump(res, open(OUT + '/results_model.json', 'w'), indent=1)
for k, v in res.items():
    print('==', k)
    print(pd.DataFrame(v).T.round(3).to_string() if isinstance(v, dict) and isinstance(next(iter(v.values())), dict) else v)
