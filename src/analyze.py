"""Produce summary tables and RQ1/RQ2/RQ3 nonparametric tests from final_scores.csv."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

KEY = ['task', 'embedding', 'classifier', 'strategy', 'seed']

def safe_wilcoxon(a, b):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = x - y
    if np.allclose(d, 0): return 1.0, 0.0
    p = wilcoxon(x, y).pvalue
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return float(p), r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='outputs/final_scores.csv')
    ap.add_argument('--output', default='outputs/analysis')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = df[(df.batch_size == 5) & np.isclose(df.budget_fraction, 0.3)].copy()
    if df.duplicated(KEY).any(): raise ValueError('Duplicate configurations found; use unique runs')
    df.groupby(['task', 'classifier']).weighted_f1.agg(['mean', 'std']).to_csv(out / 'rq1_classifier_summary.csv')
    df.groupby(['task', 'embedding']).weighted_f1.agg(['mean', 'std']).to_csv(out / 'rq2_embedding_summary.csv')
    df.groupby(['task', 'strategy']).weighted_f1.agg(['mean', 'std']).to_csv(out / 'rq3_strategy_summary.csv')
    records=[]
    for task, data in df.groupby('task'):
        # RQ1: 15 embedding x strategy blocks; average 10 seeds within block.
        tab = data.groupby(['embedding', 'strategy', 'classifier']).weighted_f1.mean().unstack('classifier').dropna()
        if {'LR','GNB'} <= set(tab.columns):
            p,r = safe_wilcoxon(tab.LR, tab.GNB)
            records.append(dict(task=task, rq='RQ1', comparison='LR vs GNB', blocks=len(tab), p_raw=p, effect_r=r, p_bonf=min(1.,4*p)))
        # RQ2: 10 classifier x strategy blocks; average seeds.
        tab = data.groupby(['classifier','strategy','embedding']).weighted_f1.mean().unstack('embedding').dropna()
        names = ['tfidf','word2vec','sbert']
        if set(names) <= set(tab.columns):
            p = friedmanchisquare(*(tab[k] for k in names)).pvalue
            records.append(dict(task=task, rq='RQ2', comparison='Friedman embeddings', blocks=len(tab), p_raw=p, effect_r=np.nan, p_bonf=np.nan))
            if p < .05:
                for i in range(3):
                    for j in range(i+1,3):
                        pv,r = safe_wilcoxon(tab[names[i]],tab[names[j]])
                        records.append(dict(task=task,rq='RQ2',comparison=f'{names[i]} vs {names[j]}',blocks=len(tab),p_raw=pv,effect_r=r,p_bonf=min(1.,3*pv)))
        # RQ3: 60 classifier x embedding x seed blocks.
        tab = data.pivot_table(index=['classifier','embedding','seed'],columns='strategy',values='weighted_f1').dropna()
        names=['random','uncertainty','qbc','information_density','contrastive']
        if set(names) <= set(tab.columns):
            p=friedmanchisquare(*(tab[k] for k in names)).pvalue
            records.append(dict(task=task,rq='RQ3',comparison='Friedman strategies',blocks=len(tab),p_raw=p,effect_r=np.nan,p_bonf=np.nan))
            if p < .05:
                for name in names[1:]:
                    pv,r=safe_wilcoxon(tab[name],tab['random'])
                    records.append(dict(task=task,rq='RQ3',comparison=f'{name} vs random',blocks=len(tab),p_raw=pv,effect_r=r,p_bonf=min(1.,4*pv)))
            paradigms=pd.DataFrame({'P1':tab[['uncertainty','qbc']].mean(axis=1),'P2':tab[['information_density','contrastive']].mean(axis=1),'RS':tab['random']})
            pp=friedmanchisquare(*(paradigms[k] for k in ['P1','P2','RS'])).pvalue
            records.append(dict(task=task,rq='RQ3',comparison='Friedman paradigms',blocks=len(tab),p_raw=pp,effect_r=np.nan,p_bonf=np.nan))
            if pp < .05:
                for left,right in [('P1','P2'),('P1','RS'),('P2','RS')]:
                    pv,r=safe_wilcoxon(paradigms[left], paradigms[right])
                    records.append(dict(task=task,rq='RQ3',comparison=f'{left} vs {right}',blocks=len(tab),p_raw=pv,effect_r=r,p_bonf=min(1.,3*pv)))
    pd.DataFrame(records).to_csv(out / 'statistical_tests.csv',index=False)
    print(f'Wrote summaries and statistical tests to {out}')

if __name__=='__main__':main()
