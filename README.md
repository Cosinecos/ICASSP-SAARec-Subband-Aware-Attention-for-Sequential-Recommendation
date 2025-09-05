# ICASSP: SAARec — Subband-Aware Attention for Sequential Recommendation

## Reviewers’ Note
Thank you for reviewing this work. Your diligence and sense of responsibility are essential to scholarly progress. Below is a concise overview of the method together with the required environment and configuration. With sincere thanks, and best wishes for your work.

---

## Method Overview (Brief)
**Motivation.** Pure time-domain self-attention often behaves like a low-pass contraction when handling abrupt, high-frequency interest shifts; frequency-domain methods can complement this but are susceptible to wavelet boundary artifacts, which may bias inference and hinder convergence. To address this, we propose **Subband-Aware Attention (SAA)**: within an invertible wavelet framework, we perform frequency-selective modeling and boundary-robust reconstruction to balance long-term preference and short-term bursts.

**Pipeline.** We first perform multi-scale wavelet decomposition on the embedded sequence, splitting it into a low-frequency approximation and multiple high-frequency subbands to obtain a joint time–frequency representation. We then apply a lightweight, single-head, shared-parameter self-attention **within each subband**, capturing dependencies per frequency band and avoiding mixing all dynamics in a single temporal channel. Next, we compute subband statistics such as energy and spectral flatness, incorporate subband hierarchy information, and feed them into a small network to obtain **cross-band weights** via softmax for adaptive fusion. During reconstruction, we introduce an overlap-add strategy and **learnable boundary tokens** (optionally with symmetric padding) before and after the inverse wavelet transform to suppress artifacts due to boundary discontinuities. Finally, we combine the reconstruction with the original sequence in a controllable low/high-frequency manner and apply residual connections and normalization. This preserves long-range trends and fine-grained short-term variations while keeping parameter and compute overhead low, and it also provides interpretability regarding band selection.

---

## Environment & Dependencies
- Python 3.10  
- PyTorch ≥ 2.3  
- torchvision  
- pytorch-wavelets == 1.3.0 (1D DWT/IWT)  
- Others: numpy ≥ 1.24, pandas ≥ 2.0, PyYAML ≥ 6.0, tqdm ≥ 4.66, matplotlib, scikit-learn

---

## Datasets & Processing
To protect our research and related rights, we currently release only the preprocessing pipeline and scripts for MovieLens-1M. The full preprocessing code for Amazon-Beauty, Amazon-Sports, and MIND-small will be open-sourced after the paper is accepted and finalized. For review and reproducibility, we provide below the unified processing rationale and key details (field specification, split strategy, output format) for all four datasets. Thank you for your understanding and support.
### 1) ML-1M (MovieLens-1M)

* **Description:** A movie recommendation benchmark with ~1M ratings, including user, movie, and timestamp.
* **Download**
  * Landing page (with documentation): <https://grouplens.org/datasets/movielens/1m/>
  * Direct link (official file directory): <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
* **Processing Steps**
  1. Group by `UserID` and sort by `Timestamp` ascending.
  2. Extract interaction sequences.
  3. **Leave-one-out:** the last item is the `target`; the rest form `seq`; write `train/val/test.jsonl`.
  4. Write `meta.json` (with `num_users` and **mapped** `num_items`).

### 2) Amazon-Beauty (All Beauty)

* **Description:** A category subset of the Amazon review data, commonly used for implicit-feedback sequence recommendation (timestamp-based).
* **Download (2018 portal):** <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>  
  In the *Per-category data* table, select **All Beauty — reviews** (typically `All_Beauty.json.gz`).
* **Fields (reviews):** `reviewerID`, `asin`, `unixReviewTime`, `overall`, etc.
* **Processing Steps**
  1. Use `reviewerID` as user and `asin` as item; sort by `unixReviewTime` ascending to build sequences.
  2. Map `asin → item_id` consecutively (start from 1).
  3. Apply leave-one-out and write `train/val/test.jsonl` and `meta.json`.

### 3) Amazon-Sports (Sports and Outdoors)

* **Description:** A sibling category subset (Sports & Outdoors) from the same Amazon data.
* **Download (2018 portal):** <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>
* **Processing Steps:** Same as Beauty (build sequences from `reviewerID/asin/unixReviewTime`; optional 5-core; leave-one-out; write JSONL and `meta.json`).

### 4) MIND-small (Microsoft News Dataset, small)

* **Description:** The small-scale version of the Microsoft news recommendation dataset; includes user behavior logs (`behaviors`) and news metadata (`news`).
* **Download**
  * Official site (license acceptance required): <https://msnews.github.io/> → *Download*
  * Small direct links (official storage container):
    * Train: `https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip`
    * Dev: `https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip`
* **Processing Steps (to the unified sequence format)**
  1. Parse `Impressions` to extract clicked `NewsID` (or directly use `History`).
  2. Group by `UserID` and sort by `Time` ascending to form click sequences.
  3. Map `NewsID → item_id` consecutively (start from 1).
  4. Apply leave-one-out and write `train/val/test.jsonl` and `meta.json`.

---

## Code Layout (Schematic)

