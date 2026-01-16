# TF-IDF Approach — Summary and Final Assessment

## Scope of This Document

This document summarizes the evolution of the **TF-IDF–based approach** for multi-class news classification, covering:

- two intermediate baselines (V1, V2),
- a final refined version (V3),
- and the rationale for concluding the TF-IDF line of experimentation.

The goal of this phase was not only to maximize performance, but also to **identify the practical and methodological limits** of classical bag-of-words representations in this task.

---

## Baseline V1 — Minimal and Fast Reference

### Motivation
Baseline V1 was designed as a **lightweight reference point**, aimed at:
- validating the signal contained in titles and metadata,
- establishing a fast and reproducible baseline,
- avoiding premature complexity.

### Key Characteristics
- **Text**: TF-IDF on titles only, minimal token filtering
- **Metadata**:
  - basic time features
  - limited source encoding (top-50 OHE)
  - raw PageRank
- **Models**:
  - Logistic Regression
  - Naive Bayes
  - Linear SVM
  - XGBoost

### Outcome
Baseline V1 already achieved a **non-trivial macro-F1 (~0.64)** on the public evaluation set, confirming that:
- titles contain meaningful information,
- classical models are viable,
- but class imbalance and confusion were already visible.

This baseline served as a **lower bound** for subsequent refinements.

---

## Baseline V2 — Structured Feature Engineering

### Motivation
Baseline V2 aimed to address the main weaknesses observed in V1 by:
- improving robustness to missing and noisy timestamps,
- increasing expressiveness of temporal features,
- normalizing informative numeric patterns in titles,
- reducing noise in ordinal metadata.

### Main Improvements
- **Time**:
  - proper timestamp parsing
  - missingness indicator
  - cyclical encodings (hour, month)
  - weekend indicator
- **Title preprocessing**:
  - HTML unescaping and Unicode normalization
  - semantic normalization of numeric patterns
  - extended n-grams (up to trigrams)
- **PageRank**:
  - ordinal compression to reduce sparsity
- **Models**:
  - focus on Linear SVM and XGBoost

### Outcome
Baseline V2 produced a **small but consistent improvement** over V1 (macro-F1 ~0.645).

However:
- gains were marginal relative to the added complexity,
- confusion matrices still showed systematic misclassification of specific classes,
- improvements mainly affected dominant classes.

This suggested **diminishing returns** from further feature engineering alone.

---

## Baseline V3 — Final TF-IDF Refinement

### Motivation
Baseline V3 represents the **most exhaustive TF-IDF configuration**, introduced to test whether remaining errors could be resolved by:
- including article text,
- fully encoding sources,
- separating preprocessing pipelines by model family,
- addressing data quality issues more aggressively.

### Key Additions
- **Article text included** with intentionally *light* preprocessing
- **Full source one-hot encoding**
- **Model-specific pipelines**:
  - Linear SVM: full sparse TF-IDF
  - XGBoost: TF-IDF + Truncated SVD
- **Explicit handling of duplicated observations**
- **Extended hyperparameter search** (especially for XGBoost)

### Outcome
Despite substantially increased modeling and preprocessing effort:

- performance gains over V2 were **negligible or negative**,
- more aggressive cleaning often **degraded results**,
- different model families converged to similar performance levels,
- training cost increased significantly without proportional benefit.

Most importantly, **error patterns remained structurally unchanged**.

---

## Cross-Version Analysis

Across V1, V2, and V3, a consistent pattern emerged:

- Performance improvements **saturated early**
- Errors were **systematic and directional**, not random
- Minority classes were repeatedly absorbed by semantically broader ones
- Model choice mattered less than representation choice

This indicates that the remaining performance gap is **not due to insufficient tuning**, but to a **representational limitation of TF-IDF** in the presence of semantic overlap between classes.

---

## Final Conclusion

The TF-IDF approach has been **thoroughly explored and exhausted**:

- it provides a strong and interpretable classical baseline,
- it establishes a clear performance reference,
- it reveals the limits of frequency-based representations for this task.

Further gains are unlikely without introducing **semantic representations** capable of modeling contextual similarity.

For these reasons, the TF-IDF approach is considered **complete and closed**, and the project transitions to an **embedding-based approach**, documented in `embedding_approach/`.

