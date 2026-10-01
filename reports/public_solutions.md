# Published Amazon ML Challenge 2026 solutions (curated 2026-10-02)

Sources found by web research (GitHub / Hugging Face; Reddit and some leaderboard pages could not be opened).
Scores are self-reported public-leaderboard numbers unless marked. No top-10 team had published a write-up.
Leaderboard top (search snippets only, unverified): CDS_Team_iisc 0.992082, GG 0.992041, 4rog 0.991762.

| Source | Claimed | Approach | Credibility |
|---|---|---|---|
| github.com/Rohilalala/amazon-ml-2026-entity-resolution | 0.989667 | rare-key IDF cosine candidates (top-10 + reverse top-3) + pretrained e5-small kNN channel (~4M extra pairs); 3 LightGBM stages (78 feats); fine-tuned e5-small pair encoder → 3-seed LightGBM stacker; one owner per record + expected-F0.5 set; encoder retrained on confident test pseudo-labels (3 rounds); France calibrated to the expected empty-entity share + legal-form / region rules | high |
| github.com/ArneshBanerjee/amazon-ml-challenge-2026-entity-resolution | 0.988216 (holdout 0.9925) | fine-tuned e5-small bi-encoder (2 rounds, hard negatives), 4 embedding views, GPU kNN both directions; LightGBM prune to ~6 candidates; two mDeBERTa-v3 cross-encoders → 2-stage LightGBM stacker with group-agreement features; isotonic calibration; expected-F0.5 top-k (empty allowed); France = intersection of two runs; 16–20 GPU h | high |
| github.com/Gauravtiwari31/Ship_It_Friday_amazon_MLC_26 | 0.987133 (rank 334) | multi-view TF-IDF + multilingual bi-encoder; LightGBM + cross-encoder cascade; one owner; France variant | medium |
| github.com/Sujal-02 (Elyptics) | 0.982 | char-ngram TF-IDF + SVD + FAISS kNN both ways + composite keys; LightGBM pre-scorer (pair recall 0.973 → 0.995); LGB/XGB/CatBoost blend + isotonic; CE on the uncertain band; prior-shift correction (2.1x near-twins in test); France pseudo-labels where two pipelines agree | high |
| github.com/SNiPERxDD/AMAZON-ML-26 | 0.980553 | 11 blocking families, LightGBM cascade, byte-level CNN two-tower, expected-F0.5; test odds ×0.47 tuned **from leaderboard feedback** | medium-high |
| github.com/ViveKumar007 | 0.977 | hashed keys + MiniLM bi-encoder, 2 LightGBM rankers, MiniLM CE; thresholds 0.75 / 0.97 (France) | high |
| github.com/RahulBharadwaz/Amazon_ML_Challenge_2026 | val 0.988 on a 6 % sample (no LB) | the only HNSW solution found (int8 HNSW + TF-IDF + number hashing, 2-stage LightGBM) | the "0.990 with HNSW" claim is unverified |

Common to the 0.985+ solutions: (1) sparse rare-token keys **plus a fine-tuned dense bi-encoder kNN channel in both
directions**; (2) LightGBM stacked with fine-tuned multilingual cross-encoder scores (e5 / mDeBERTa) and
within-entity agreement features; (3) one owner per record + calibrated expected-F0.5 sets; (4) France: stricter /
agreement-based rules, sometimes test pseudo-labels.

Rule-risk notes: the official data is re-hosted publicly (HF / Kaggle); leaderboard-tuned odds (SNiPERxDD) and test
pseudo-labelling are transductive / leaderboard probing — we use neither.

Our position (E046): we have (2), (3) and the sparse half of (1). Missing: the dense bi-encoder channel → E047.
Retrieval loss on our holdout = 0.0059 F0.5 (11,227 of 623,372 true matches never retrieved).
