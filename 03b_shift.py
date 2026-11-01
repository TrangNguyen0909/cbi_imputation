"""Which features drive the residual public-vs-CBI shift? Ablation of adversarial AUC."""
import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
exec(open('03_mask_aug.py').read().split("D = X[X.status == 'public']")[0])  # reuse setup (masks, prep, FEAT)
D = X[X.status == 'public'].reset_index(drop=True); C = X[X.status == 'CBI']
Dm = apply_mask(D, sample_masks(len(D)))
A = prep(pd.concat([Dm[FEAT], C[FEAT]], ignore_index=True)); lab = np.r_[np.zeros(len(Dm)), np.ones(len(C))]
def auc(cols):
    oof = np.zeros(len(A))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=1).split(A, lab):
        oof[te] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1).fit(A.iloc[tr][cols], lab[tr]).predict_proba(A.iloc[te][cols])[:, 1]
    return roc_auc_score(lab, oof)
FA = [f for f in FEAT if f != 'frac_cbi_chem_others']
print('all', round(auc(FA), 3))
for f in FA: print(f'single {f:22s}', round(auc([f]), 3))
