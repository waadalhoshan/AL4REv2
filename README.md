# Active Learning for Requirements Classification — Reproduction Kit

Companion experiment runner for **“Active Learning for Requirements Classification: An Empirical Study”** by Shatha Azzaben and Waad Alhoshan.

> **Important status:** This is an **independent implementation of the protocol described in the manuscript**.

## What it reproduces

| Dimension | Settings |
|---|---|
| Tasks | T1 Dalpiaz functional/NF (956); T2 SecReq security (510); T3 PROMISE 10 NFR classes (370); T4 top four NFR classes (249) |
| Models | Logistic Regression (`LR`), Gaussian Naïve Bayes (`GNB`) |
| Embeddings | TF-IDF unigram/bigram, Google News Word2Vec 300d averaged, SBERT |
| Queries | Random (`random`), entropy Uncertainty (`uncertainty`), bagged vote-entropy QbC (`qbc`), entropy × kNN mean cosine density (`information_density`), entropy + novelty (`contrastive`) |
| Seeds | 42–51 |
| Default | One stratified labeled seed per class; batch 5; 30% of remaining training pool, floor to full batches |
| Sensitivity | Batch sizes 5/10 × budgets 30%/40% |
| Metrics | Weighted F1 (primary), macro F1, weighted precision/recall, accuracy |
| Statistical analysis | Matched-condition Wilcoxon and Friedman with Bonferroni correction |

The default full factorial design is **4 × 3 × 2 × 5 × 10 = 1,200 experimental runs**. SBERT/Word2Vec downloads and the full matrix can require substantial computing resources.

## 1. Set up

```bash
git clone https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Run commands from this package’s root (where `configs/` resides). For a standalone copy, skip clone and `cd` into the extracted folder.

## 2. Use datasets already in your repository

Edit `configs/experiments.yaml` to point each task to an existing CSV or XLSX and identify its **text** and **label** columns. Expected simplified table:

```csv
text,label
"The application shall support login",Functional
"Access shall be logged",Non-functional
```

**Do not overwrite original datasets.** The same PROMISE NFR file is used for T3 and T4: T3 excludes the one-instance `Portability` class and T4 keeps `Usability`, `Security`, `Operational`, `Performance`. Verify your repository’s labels and the expected row counts before running. Original labels might use codes (e.g., `US`, `SE`, `O`, `PE`), in which case edit `include_labels` appropriately.

Word2Vec uses the original Google News 300-dimensional vectors, which **are not bundled**. Download these from an authorized source and set `models.word2vec_path` in the YAML. The runner will fail with a clear message if that model is missing. SBERT automatically downloads the configured sentence encoder on first use; cache the pretrained model for offline runs.

## 3. Run experiments

**Quick smoke run (one experiment):**

```bash
python -m src.run_experiments --task T1 --embedding tfidf --classifier LR --strategy uncertainty --seed 42
```

**Entire default experiment grid:**

```bash
python -m src.run_experiments
```

**All sensitivity combinations (includes the default setting):**

```bash
python -m src.run_experiments --sensitivity
```

**Single task or model:**

```bash
python -m src.run_experiments --task T3 --embedding sbert --classifier GNB
```

The runner writes `outputs/learning_curves.csv` (each evaluation checkpoint), `outputs/final_scores.csv` (final checkpoint of each condition), and `outputs/run_config.json` (settings snapshot). Running again **overwrites** output files; use separate `output_dir` values for separate campaigns. Runs are evaluated at the initial seed point and after each acquisition batch.

## 4. Recompute paper-style analysis

```bash
python -m src.analyze --input outputs/final_scores.csv
```

Creates `outputs/analysis/` with task-level summaries and inferential tests:
- `rq1_classifier_summary.csv` and Wilcoxon by 15 paired embedding × strategy blocks (seed means)
- `rq2_embedding_summary.csv` and Friedman by 10 classifier × strategy blocks (seed means), then pairwise Wilcoxon
- `rq3_strategy_summary.csv` and Friedman by 60 classifier × embedding × seed blocks, plus comparisons to random and P1/P2/RS
- `statistical_tests.csv` with p-values and rank-biserial effect sizes

For meaningful inference, first run the **full default 1,200-condition grid**. Small smoke runs will not contain enough matched blocks.

## 5. Validate and compare with uploaded results

```bash
pytest -q
```

Compare `outputs/final_scores.csv` with the **existing results on GitHub** by matching task, classifier, embedding, strategy, seed, batch size and budget. Check dataset row counts, label frequencies, train/test division, initial seed selection and dependency/model versions before interpreting score differences. Do **not** assume the newly computed values must match paper tables exactly until the open implementation choices below are resolved.

## Reproducibility details requiring confirmation


| Choice | Currently used | 
|---|---|---|
| Holdout split | Stratified 80/20, split independently by seed | 
|SBERT checkpoint | `all-MiniLM-L6-v2` |
| QbC committee size | 5 stratified bootstrap members | 
| Density neighbors | 10, cosine similarity | 
| Contrastive rule | Normalized 50/50 entropy and distance-to-nearest-labeled | 
| Word tokenization | `lower().split()` and mean of in-vocabulary vectors | 
| LR regularization, convergence | scikit-learn defaults, `max_iter=1000` | 
| Budget interpretation | Floor `(0.30 × unlabeled_training_pool)/batch` full batches | 
| Train/test representation | TF-IDF fit only on train; embeddings applied to train/test | 

## Directory map

```text
configs/experiments.yaml       Dataset paths, model settings, experiment grid
src/run_experiments.py         Embedding, classifier, acquisition and AL loop
src/analyze.py                 Summaries, matched statistics
requirements.txt               Python dependencies
tests/                          Sanity tests
outputs/                        Generated locally; git-ignored
```

## Provenance and citation

Please cite the companion paper in any derivative study. Datasets remain subject to their respective source terms and attribution requirements. This kit does not contain the datasets, stored results, pretrained Google News vectors, or a claim of exact reproducibility.

### Curret Reference for the study
Azzaben, S., & Alhoshan, W. (2026).
*Active Learning for Requirements Classification: An Empirical Study.*
