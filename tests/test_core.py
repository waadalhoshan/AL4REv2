import numpy as np
from src.run_experiments import entropy, metrics, query
from sklearn.linear_model import LogisticRegression

def test_entropy_uniform_exceeds_confident():
    assert entropy(np.array([[0.5, .5]]))[0] > entropy(np.array([[.99, .01]]))[0]

def test_metrics_perfect():
    assert metrics(np.array(['a','b']),np.array(['a','b']))['weighted_f1'] == 1

def test_query_unique():
    x=np.array([[0.,1.],[1.,0.],[.2,.8],[.8,.2],[.4,.6]])
    y=np.array(['a','b'])
    model=LogisticRegression().fit(x[:2],y)
    cfg={'protocol':{'committee_size':3,'density_k':2,'contrastive_weight':.5,'lr_max_iter':1000}}
    for strategy in ['random','uncertainty','qbc','information_density','contrastive']:
        sel=query(strategy,model,x[2:],x[:2],y,2,np.random.default_rng(42),'LR',42,cfg)
        assert len(sel)==2 and len(set(sel))==2 and max(sel)<3
