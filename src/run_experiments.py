"""Re-runnable active learning benchmark implementing the paper's stated protocol.

Unspecified implementation details are explicitly marked as assumptions in YAML.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity


def load_task(spec):
    p = Path(spec['path'])
    if not p.exists():
        raise FileNotFoundError(f'Missing dataset: {p}. Edit configs/experiments.yaml to point to your existing GitHub data.')
    df = pd.read_csv(p) if p.suffix.lower() == '.csv' else pd.read_excel(p)
    t, y = spec.get('text_column', 'text'), spec.get('label_column', 'label')
    if t not in df or y not in df:
        raise ValueError(f'{p}: expected columns {t!r}, {y!r}; found {list(df.columns)}')
    df = df[[t, y]].dropna().copy()
    df[t] = df[t].astype(str)
    df[y] = df[y].astype(str)
    if 'exclude_labels' in spec:
        df = df[~df[y].isin(map(str, spec['exclude_labels']))]
    if 'include_labels' in spec:
        df = df[df[y].isin(map(str, spec['include_labels']))]
    if df[y].nunique() < 2:
        raise ValueError(f'{p}: need >=2 classes after filtering')
    return df[t].tolist(), df[y].to_numpy()


def embed(train_texts, test_texts, kind, cfg):
    # Fit TF-IDF on training data only to prevent test-set leakage.
    if kind == 'tfidf':
        v = TfidfVectorizer(ngram_range=(1, 2))
        return v.fit_transform(train_texts).toarray(), v.transform(test_texts).toarray()
    if kind == 'sbert':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg['models']['sbert'])
        return (model.encode(train_texts, convert_to_numpy=True, show_progress_bar=False),
                model.encode(test_texts, convert_to_numpy=True, show_progress_bar=False))
    if kind == 'word2vec':
        from gensim.models import KeyedVectors
        path = Path(cfg['models']['word2vec_path'])
        if not path.exists():
            raise FileNotFoundError(f'Word2Vec model missing: {path}; see README')
        kv = KeyedVectors.load_word2vec_format(str(path), binary=path.suffix == '.bin')
        def convert(texts):
            vectors = []
            for text in texts:
                vs = [kv[w] for w in text.lower().split() if w in kv]
                vectors.append(np.mean(vs, axis=0) if vs else np.zeros(kv.vector_size))
            return np.asarray(vectors)
        return convert(train_texts), convert(test_texts)
    raise ValueError(kind)


def learner(name, seed, cfg):
    if name == 'LR':
        return LogisticRegression(max_iter=cfg['protocol'].get('lr_max_iter', 1000), random_state=seed)
    if name == 'GNB':
        return GaussianNB()
    raise ValueError(name)


def entropy(prob):
    p = np.clip(prob, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1)


def query(strategy, model, x_pool, x_labeled, y_labeled, batch, rng, learner_name, seed, cfg):
    n = len(x_pool)
    if strategy == 'random':
        return rng.choice(n, size=min(batch, n), replace=False)
    prob = model.predict_proba(x_pool)
    e = entropy(prob)
    if strategy == 'uncertainty':
        score = e
    elif strategy == 'qbc':
        votes = []
        classes = model.classes_
        for i in range(cfg['protocol']['committee_size']):
            # Stratified bootstrap guarantees at least one labeled sample from each class.
            idx = np.concatenate([rng.choice(np.where(y_labeled == c)[0],
                                              size=np.sum(y_labeled == c), replace=True)
                                  for c in classes])
            member = learner(learner_name, seed + i + 1, cfg)
            member.fit(x_labeled[idx], y_labeled[idx])
            votes.append(member.predict(x_pool))
        vote_matrix = np.asarray(votes).T
        counts = np.stack([(vote_matrix == c).mean(axis=1) for c in classes], axis=1)
        score = entropy(counts)
    elif strategy == 'information_density':
        sim = cosine_similarity(x_pool)
        np.fill_diagonal(sim, -np.inf)
        k = min(cfg['protocol']['density_k'], max(1, n - 1))
        if n == 1:
            density = np.ones(1)
        else:
            density = np.sort(sim, axis=1)[:, -k:].mean(axis=1)
        score = e * np.maximum(density, 0)
    elif strategy == 'contrastive':
        dist_to_labeled = 1 - cosine_similarity(x_pool, x_labeled).max(axis=1)
        weight = cfg['protocol']['contrastive_weight']
        def norm(a):
            return (a - a.min()) / (np.ptp(a) + 1e-12)
        score = (1 - weight) * norm(e) + weight * norm(dist_to_labeled)
    else:
        raise ValueError(strategy)
    # Seed-controlled tie-breaking; no instance is queried twice.
    return np.lexsort((rng.random(n), -score))[:min(batch, n)]


def metrics(y_true, predicted):
    return dict(weighted_f1=f1_score(y_true, predicted, average='weighted', zero_division=0),
                macro_f1=f1_score(y_true, predicted, average='macro', zero_division=0),
                weighted_precision=precision_score(y_true, predicted, average='weighted', zero_division=0),
                weighted_recall=recall_score(y_true, predicted, average='weighted', zero_division=0),
                accuracy=accuracy_score(y_true, predicted))


def run_one(task, texts, labels, emb, classifier, strategy, seed, cfg, batch, budget):
    p = cfg['protocol']
    indices = np.arange(len(labels))
    tr, te = train_test_split(indices, test_size=p['test_fraction'], random_state=seed, stratify=labels)
    xtrain, xtest = embed([texts[i] for i in tr], [texts[i] for i in te], emb, cfg)
    ytrain, ytest = labels[tr], labels[te]
    rng = np.random.default_rng(seed)
    chosen = np.array([rng.choice(np.where(ytrain == c)[0]) for c in np.unique(ytrain)], dtype=int)
    available = np.setdiff1d(np.arange(len(tr)), chosen)
    # B applies to the pool AFTER selecting the stratified seed set, as described.
    max_acquisitions = int(math.floor(budget * len(available) / batch)) * batch
    history = []
    total = 0
    while True:
        model = learner(classifier, seed, cfg)
        model.fit(xtrain[chosen], ytrain[chosen])
        scores = metrics(ytest, model.predict(xtest))
        history.append(dict(task=task, embedding=emb, classifier=classifier,
                            strategy=strategy, seed=seed, batch_size=batch,
                            budget_fraction=budget, iteration=len(history),
                            acquired=total, labeled=len(chosen), **scores))
        if total >= max_acquisitions or not len(available):
            break
        local = query(strategy, model, xtrain[available], xtrain[chosen], ytrain[chosen],
                      min(batch, max_acquisitions - total), rng, classifier, seed, cfg)
        selected = available[local]
        chosen = np.concatenate([chosen, selected])
        available = np.setdiff1d(available, selected)
        total += len(selected)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments.yaml')
    parser.add_argument('--task', default='all', help='T1, T2, T3, T4 or all')
    parser.add_argument('--embedding', default='all')
    parser.add_argument('--classifier', default='all')
    parser.add_argument('--strategy', default='all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sensitivity', action='store_true', help='Run 5/10 batch x 30/40 percent budget (deduplicates default)')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg.get('output_dir', 'outputs'))
    out.mkdir(parents=True, exist_ok=True)
    def restrict(items, wanted):
        if wanted == 'all': return items
        if wanted not in items: parser.error(f'{wanted!r} not in {items}')
        return [wanted]
    tasks = restrict(list(cfg['datasets']), args.task)
    emb = restrict(cfg['embeddings'], args.embedding)
    cls = restrict(cfg['classifiers'], args.classifier)
    strat = restrict(cfg['strategies'], args.strategy)
    seeds = [args.seed] if args.seed is not None else cfg['seeds']
    protocols = [(5, 0.3), (5, 0.4), (10, 0.3), (10, 0.4)] if args.sensitivity else [(cfg['protocol']['batch_size'], cfg['protocol']['budget_fraction'])]
    rows = []
    for task in tasks:
        texts, labels = load_task(cfg['datasets'][task])
        print(f'{task}: {len(texts)} records, {len(set(labels))} labels', flush=True)
        for e, c, q, seed, (b, budget) in itertools.product(emb, cls, strat, seeds, protocols):
            print(f'  {e} {c} {q} seed={seed} batch={b} budget={budget}', flush=True)
            rows.extend(run_one(task, texts, labels, e, c, q, seed, cfg, b, budget))
            # Append completed condition immediately; resumable via external output management.
            pd.DataFrame(rows).to_csv(out / 'learning_curves.csv', index=False)
    df = pd.DataFrame(rows)
    df.sort_values('iteration').groupby(['task', 'embedding', 'classifier', 'strategy', 'seed', 'batch_size', 'budget_fraction'], as_index=False).tail(1).to_csv(out / 'final_scores.csv', index=False)
    (out / 'run_config.json').write_text(json.dumps(cfg, indent=2))
    print(f'Done: {len(df)} evaluation checkpoints saved to {out}')

if __name__ == '__main__':
    main()
