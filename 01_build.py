"""Build site-level modelling table from CDR 2024 public files."""
import pandas as pd, numpy as np, re, sys
D, OUT = sys.argv[1], sys.argv[2]
R = lambda f: pd.read_csv(f'{D}/2024 CDR {f}.csv', encoding='latin1', low_memory=False)
m, i, c, n = R('Manufacture-Import Information'), R('Industrial Processing and Use Information'), R('Consumer and Commercial Use Information'), R('Nationally Aggregated Production Volumes')
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
def rng(s):
    s = str(s); v = [float(x.replace(',', '')) for x in re.findall(r'[\d,]+(?:\.\d+)?', s) if x.replace(',', '')]
    if not v: return (np.nan, np.nan)
    if s.strip().startswith('<'): return (0.0, v[0])
    return (v[0], v[-1])
key = ['CHEMICAL ID', 'EPA-TSCA PROGRAM ID']
m['row_id'] = np.arange(len(m))
m['pv'] = num(m['2023 PV'])
m['status'] = np.where(m['pv'].notna(), 'public', np.where(m['2023 PV'].astype(str) == 'CBI', 'CBI', 'other'))
# national ranges
n['nat_lo'], n['nat_hi'] = zip(*n['2023 NATIONALLY AGGREGATED PV'].map(rng))
nat = n.drop_duplicates('CHEMICAL ID').set_index('CHEMICAL ID')[['nat_lo', 'nat_hi', 'CHEMICAL ID TYPE']]
X = m[key + ['row_id', 'status', 'pv']].copy()
cat_src = {'activity': 'ACTIVITY', 'state': 'SITE STATE', 'workers': 'WORKERS CODE', 'maxconc': 'MAX CONC CODE', 'byp': 'PCT BYP CODE',
           'recycled': 'RECYCLED OR OTHERWISE USED', 'never_at_site': 'CHEM NEVER AT SITE', 'foreign_cc': 'FOREIGN PC COUNTRY CODE', 'phys_form': 'PHYSICAL FORM(S) LIST'}
for k, v in cat_src.items(): X[k] = m[v].astype(str)
X['naics6'] = m['SITE NAICS CODE 1'].astype(str).str[:6]; X['naics3'] = X['naics6'].str[:3]
X['parent'] = m['STANDARDIZED PARENT COMPANY NAME'].astype(str)
X['id_type'] = m['CHEMICAL ID TYPE'].astype(str)
X['n_cbi_fields'] = (m.astype(str) == 'CBI').sum(axis=1)
for k, v in {'onsite': '2023 V USED ON-SITE', 'exported': '2023 V EXPORTED'}.items():
    X['log_' + k] = np.log10(num(m[v]) + 1); X[k + '_cbi'] = (m[v].astype(str) == 'CBI').astype(int)
X = X.join(nat[['nat_lo', 'nat_hi']], on='CHEMICAL ID')
X['nat_log_lo'] = np.log10(X.nat_lo + 1); X['nat_log_hi'] = np.log10(X.nat_hi + 1)
X['n_sites_chem'] = X.groupby('CHEMICAL ID').row_id.transform('count')
X['nat_log_per_site'] = np.log10((np.sqrt(np.maximum(X.nat_lo, 1) * X.nat_hi)) / X.n_sites_chem + 1)
X['n_chem_site'] = X.groupby('EPA-TSCA PROGRAM ID').row_id.transform('count')
X['n_rows_parent'] = X.groupby('parent').row_id.transform('count')
X['frac_cbi_chem_others'] = X.groupby('CHEMICAL ID').status.transform(lambda s: (s == 'CBI').mean())
ind = i.groupby(key).agg(n_ind=('IND FC CODE', 'count'), n_ind_sect=('IND SECT CODE', 'nunique'), ind_fc=('IND FC CODE', 'first'),
                         ind_sect=('IND SECT CODE', 'first'), ind_type=('IND TYPE CODE', 'first'), ind_sites=('IND SITES CODE', 'first'),
                         ind_workers=('IND WORKERS CODE', 'first')).reset_index()
con = c.groupby(key).agg(n_con=('C / C PROD CAT CODE', 'count'), con_cat=('C / C PROD CAT CODE', 'first'), con_use=('CONS AND/OR COMM USE', 'first'),
                         kids=('USED IN PROD FOR CHILDREN', 'first')).reset_index()
X = X.merge(ind, on=key, how='left').merge(con, on=key, how='left')
X[['n_ind', 'n_ind_sect', 'n_con']] = X[['n_ind', 'n_ind_sect', 'n_con']].fillna(0)
X['y'] = np.log10(X.pv.clip(lower=1))
X.to_parquet(OUT + '/site_table.parquet', index=False)
print(X.status.value_counts().to_dict())
print('other rows sample PV values:', m.loc[X.status == 'other', '2023 PV'].astype(str).value_counts().head(3).to_dict())
# sanity: chemicals whose every site is public -> sum vs national range
g = X.groupby('CHEMICAL ID').agg(allpub=('status', lambda s: (s == 'public').all()), s=('pv', 'sum'), lo=('nat_lo', 'first'), hi=('nat_hi', 'first'))
a = g[g.allpub & g.hi.notna()]
print('chems all-public:', len(a), ' sum within national range:', round(((a.s >= a.lo * 0.999) & (a.s <= a.hi * 1.001)).mean(), 3))
cb = X[X.status == 'CBI']
print('CBI rows whose chemical has >=1 public site:', round(cb['CHEMICAL ID'].isin(X.loc[X.status == 'public', 'CHEMICAL ID']).mean(), 3))
