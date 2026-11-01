"""Empirical CBI mask-pattern augmentation, simulated-CBI evaluation, residual shift, conformalized quantile regression."""
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT = sys.argv[1]; rng = np.random.default_rng(0)
X = pd.read_parquet(OUT + '/site_table.parquet'); X = X[X.status.isin(['public', 'CBI'])].reset_index(drop=True)
raw_site_cbi = X['EPA-TSCA PROGRAM ID'].astype(str) == 'CBI'; raw_parent_cbi = X['parent'].astype(str) == 'CBI'
X.loc[raw_site_cbi, 'n_chem_site'] = np.nan; X.loc[raw_parent_cbi, 'n_rows_parent'] = np.nan
X['nat_log_per_site'] = X['nat_log_per_site'].fillna(X['nat_log_per_site'].median())
CAT = ['activity','state','workers','maxconc','byp','recycled','never_at_site','foreign_cc','phys_form','naics6','naics3','id_type',
       'ind_fc','ind_sect','ind_type','ind_sites','ind_workers','con_cat','con_use','kids']
NUM = ['log_onsite','log_exported','nat_log_lo','nat_log_hi','nat_log_per_site','n_sites_chem','n_chem_site','n_rows_parent',
       'frac_cbi_chem_others','n_ind','n_ind_sect','n_con']
FEAT = CAT + NUM
# mask groups: name -> (indicator from raw, columns affected)
MG = {'site': (raw_site_cbi | (X.state.astype(str) == 'CBI'), ['state','naics6','naics3','n_chem_site']),
      'parent': (raw_parent_cbi | (X.foreign_cc.astype(str) == 'CBI'), ['foreign_cc','n_rows_parent']),
      'activity': (X.activity.astype(str) == 'CBI', ['activity']), 'workers': (X.workers.astype(str) == 'CBI', ['workers']),
      'maxconc': (X.maxconc.astype(str) == 'CBI', ['maxconc']), 'byp': (X.byp.astype(str) == 'CBI', ['byp']),
      'recycled': (X.recycled.astype(str) == 'CBI', ['recycled']), 'phys_form': (X.phys_form.astype(str) == 'CBI', ['phys_form']),
      'onsite': (X.onsite_cbi == 1, ['log_onsite']), 'exported': (X.exported_cbi == 1, ['log_exported'])}
M = pd.DataFrame({k: v[0].values for k, v in MG.items()})
cbi_patterns = M[X.status.values == 'CBI'].values
def apply_mask(df, pats):
    df = df.copy()
    for j, (k, (_, cols)) in enumerate(MG.items()):
        sel = pats[:, j]
        for c in cols:
            if c in CAT: df[c] = df[c].astype(object); df.loc[sel, c] = 'CBI'
            else: df.loc[sel, c] = np.nan
    return df
def sample_masks(n): return cbi_patterns[rng.integers(0, len(cbi_patterns), n)]
def prep(df):
    df = df.copy()
    for c in CAT: df[c] = pd.Categorical(df[c].astype(object).fillna('NA').astype(str), categories=CATS[c])
    return df
CATS = {c: sorted(set(X[c].astype(object).fillna('NA').astype(str)) | {'CBI'}) for c in CAT}
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, cat_smooth=20, verbose=-1)
BINS = [np.log10(25e3), 5, 6, 7, 8]; b = lambda v: np.digitize(v, BINS)
def metrics(yt, yp): return dict(MAE=float(np.mean(abs(yt-yp))), RMSE=float(np.sqrt(np.mean((yt-yp)**2))), bin_acc=float(np.mean(b(yt)==b(yp))),
                                 within1=float(np.mean(abs(b(yt)-b(yp))<=1)), spearman=float(spearmanr(yt, yp)[0]))
def augment(df, k=2):
    return pd.concat([df] + [apply_mask(df, sample_masks(len(df))) for _ in range(k)], ignore_index=True)
D = X[X.status == 'public'].reset_index(drop=True); y = D.y.values
res = {}
# residual shift after harmonizing masks
Dm = apply_mask(D, sample_masks(len(D))); C = X[X.status == 'CBI']
A = prep(pd.concat([Dm[FEAT], C[FEAT]], ignore_index=True)); lab = np.r_[np.zeros(len(Dm)), np.ones(len(C))]
oof = np.zeros(len(A))
FA = [f for f in FEAT if f != 'frac_cbi_chem_others']
for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(A, lab):
    oof[te] = lgb.LGBMClassifier(**{**P, 'n_estimators': 300}).fit(A.iloc[tr][FA], lab[tr]).predict_proba(A.iloc[te][FA])[:, 1]
res['adversarial_auc_after_mask_harmonization'] = float(roc_auc_score(lab, oof))
pc = np.clip(oof[:len(Dm)], 0.01, 0.99); w_pub = pc / (1 - pc)   # density-ratio weights for public rows
# CV: unseen chemicals, test rows masked with empirical CBI patterns
out = {k: np.zeros(len(D)) for k in ['clean_model', 'augmented_model', 'equal_split', 'q05', 'q95']}
test_masked = np.zeros(len(D), bool)
gkf = list(GroupKFold(5).split(D, y, D['CHEMICAL ID']))
for f, (tr, te) in enumerate(gkf):
    Te = prep(apply_mask(D.loc[te], sample_masks(len(te))))
    m0 = lgb.LGBMRegressor(**P).fit(prep(D.loc[tr])[FEAT], y[tr]); out['clean_model'][te] = m0.predict(Te[FEAT])
    Tr = augment(D.loc[tr]); yt = np.tile(y[tr], 3)
    m1 = lgb.LGBMRegressor(**P).fit(prep(Tr)[FEAT], yt); out['augmented_model'][te] = m1.predict(Te[FEAT])
    for q, nm in [(0.05, 'q05'), (0.95, 'q95')]:
        out[nm][te] = lgb.LGBMRegressor(**{**P, 'objective': 'quantile', 'alpha': q}).fit(prep(Tr)[FEAT], yt).predict(Te[FEAT])
    out['equal_split'][te] = D.loc[te, 'nat_log_per_site'].values
res['simulated_CBI_test_unseen_chemical'] = {k: metrics(y, out[k]) for k in ['equal_split', 'clean_model', 'augmented_model']}
# CQR: calibrate on half of chemicals, evaluate on other half (repeat 20 splits)
chem = D['CHEMICAL ID'].values; uc = np.unique(chem)
s = np.maximum(out['q05'] - y, y - out['q95'])
cov, covw, wid, widw, cov_hi, covw_hi = [], [], [], [], [], []
hi = pc >= np.quantile(pc, 0.8)
for r in range(20):
    cal = np.isin(chem, rng.choice(uc, len(uc) // 2, replace=False)); ev = ~cal
    qh = np.quantile(s[cal], np.ceil((cal.sum() + 1) * 0.9) / cal.sum())
    # weighted conformal (Tibshirani et al. 2019): weighted quantile of calibration scores
    o = np.argsort(s[cal]); sw = s[cal][o]; ww = w_pub[cal][o]; cw = np.cumsum(ww) / (ww.sum() + np.median(w_pub[ev]))
    qw = sw[np.searchsorted(cw, 0.9)] if (cw >= 0.9).any() else np.inf
    lo, up = out['q05'] - qh, out['q95'] + qh; low, upw = out['q05'] - qw, out['q95'] + qw
    inn = (y >= lo) & (y <= up); innw = (y >= low) & (y <= upw)
    cov.append(inn[ev].mean()); covw.append(np.average(innw[ev], weights=w_pub[ev])); wid.append((up - lo)[ev].mean()); widw.append((upw - low)[ev].mean())
    cov_hi.append(inn[ev & hi].mean()); covw_hi.append(innw[ev & hi].mean())
res['conformal_90'] = dict(raw_quantile_coverage=float(np.mean((y >= out['q05']) & (y <= out['q95']))),
    CQR_coverage=float(np.mean(cov)), CQR_mean_width_log10=float(np.mean(wid)), CQR_coverage_CBIlike_top20=float(np.mean(cov_hi)),
    weightedCQR_coverage_shift_weighted=float(np.mean(covw)), weightedCQR_mean_width_log10=float(np.mean(widw)), weightedCQR_coverage_CBIlike_top20=float(np.mean(covw_hi)))
res['simulated_CBI_test_CBIlike_top20'] = {k: metrics(y[hi], out[k][hi]) for k in ['equal_split', 'clean_model', 'augmented_model']}
D['oof_aug'] = out['augmented_model']; D['oof_q05'] = out['q05']; D['oof_q95'] = out['q95']; D['w_shift'] = w_pub
D.to_parquet(OUT + '/public_oof_masked.parquet', index=False)
json.dump(res, open(OUT + '/results_mask_aug.json', 'w'), indent=1)
for k, v in res.items():
    print('==', k); print(pd.DataFrame(v).T.round(3).to_string() if isinstance(v, dict) and isinstance(next(iter(v.values())), dict) else v)
