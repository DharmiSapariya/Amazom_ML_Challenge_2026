<div align="center">

# 🏆 Amazon ML Challenge 2026
### Noisy Business Entity Resolution Pipeline

**Clean → Shortlist → Verify → Match**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![RapidFuzz](https://img.shields.io/badge/RapidFuzz-string%20matching-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-competition-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

| 🎯 Blocking recall | 🔗 True links found | 🧮 Entities validated | 🧪 Strategies tried |
|:---:|:---:|:---:|:---:|
| **92.35%** | **159,970 / 173,213** | **49,996** | **8** |

</div>

> **Note:** 92.35% is **blocking recall** (the ceiling on what any matcher can reach) on the validated 49,996-entity sample. It is **not** the final competition score.

---

## ⚡ The Whole Project in 30 Seconds

```mermaid
flowchart LR
    A["😵 Messy business<br/>records"] --> B["🧹 Clean them"]
    B --> C["🔎 Shortlist likely<br/>matches"]
    C --> D["📏 Measure what<br/>the shortlist missed"]
    D --> E["🤖 Decide: same<br/>business or not?"]
    E --> F["✅ Final matches"]
```

| Step | In plain words |
|---|---|
| 🧹 Clean | Make names and addresses comparable (case, accents, punctuation) |
| 🔎 Shortlist | Don't compare everything with everything. Pull a small list of *possible* matches per business |
| 📏 Measure | Check the shortlist against the answer key to see how many real matches survive |
| 🤖 Decide | For each shortlisted pair, a model says how likely it is the same business |
| ✅ Output | Keep only confident pairs. Leaving a business with *no* match is allowed |

---

## 🎯 The Problem

There is one clean reference database (**S1**) and two noisy ones (**S2**, **S3**). For every business in S1, find its matches in S2 and S3.

```mermaid
flowchart TD
    S1["S1: reference DB<br/>(clean, deduplicated)"] -->|find matches in| S2["S2: noisy DB"]
    S1 -->|find matches in| S3["S3: noisy DB"]
    S2 --> R{"Result for each S1 business"}
    S3 --> R
    R --> Z["0 matches ✅ valid"]
    R --> O["1 match"]
    R --> M["Many matches"]
```

⚠️ Matching is **not** one-to-one.

**Why it's hard.** The same business shows up in many disguises:

| Clean (S1) | Noisy (S2 / S3) | What changed |
|---|---|---|
| Primary Care Specialists Inc | Primary Care Specialists Incorporated | Legal suffix |
| Primary Care Specialists Inc | primary care stpianlsts inc | Typo + case |
| Straight Edge Hypnosis | Straight Edge Hpyosris | Typos |

Addresses add abbreviations, missing or reordered parts, transliteration and landmarks on top.

| Approach | Works? |
|---|:---:|
| Exact match | ❌ |
| Normalization alone | ❌ |
| One giant fuzzy search over everything | ❌ (millions × millions) |
| **Two stages: shortlist first, then verify** | ✅ |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A["Raw S1 / S2 / S3"] --> B["Chunked loading"]
    B --> C["Normalization"]
    C --> D["C9 multi-key blocking"]
    D --> E["Rank by shared keys"]
    E --> F["Candidate pairs"]
    F --> G["Similarity features"]
    G --> H["Hard negatives"]
    H --> I["Matching model"]
    I --> J["P(match)"]
    J --> K["Entity-level threshold"]
    K --> L["Final results"]

    classDef a fill:#d4edda,stroke:#28a745,color:#000;
    classDef b fill:#d6e9ff,stroke:#2f80ed,color:#000;
    class A,B,C,D,E,F a;
    class G,H,I,J,K,L b;
```

🟩 **Stage A: Retrieval / Blocking** (find plausible candidates, maximize recall)
🟦 **Stage B: Matching** (decide which candidates are truly the same business, maximize precision)

**Why two stages?** The matcher can only judge pairs it is given, so blocking recall is a hard ceiling:

```mermaid
flowchart LR
    A["Blocking misses a<br/>true pair"] --> B["Matcher never sees it"] --> C["Lost for good ❌"]
    D["Blocking keeps it"] --> E["Matcher can find it"] --> F["Can be matched ✅"]
```

---

## 🔄 Pipeline Interaction Diagram

Every component, what it receives and what it hands over:

```mermaid
sequenceDiagram
    autonumber
    participant Raw as Raw S1 / S2 / S3
    participant Norm as Normalizer (chunked)
    participant C9 as C9 Multi-Key Blocker
    participant Val as Recall Validator
    participant Fx as Missed-Pair Forensics
    participant Feat as Feature Builder
    participant Model as Matcher (GBM)
    participant Thr as Entity-Level Threshold
    participant Out as matching_results.tsv

    Raw->>Norm: stream 100k-row chunks
    Note over Norm: lowercase · NFKD · strip accents<br/>& → and · drop punctuation · collapse spaces
    Norm->>C9: normalized names + addresses
    Note over C9: 8 keys: NT NP5 NP6 NE NS AT AP6 AN<br/>country-scoped · stopword-filtered<br/>max key freq 1200 · max 500 cand per S1
    C9->>Val: 24,104,327 candidate pairs
    Val->>Val: compare to 173,213 ground-truth links
    Val-->>C9: 92.35% recall (13,243 missed)
    Val->>Fx: missed pairs
    Fx-->>Feat: use NAME + ADDRESS + MISSINGNESS (98.65% recoverable)
    C9->>Feat: candidate pairs + blocking signals
    Feat->>Model: similarity · structural · blocking features
    Model->>Thr: P(match) per pair
    Thr->>Out: keep confident pairs per S1 (empty rows allowed)
```

---

## 📉 The Funnel: From Millions of Comparisons to a Usable Shortlist

```mermaid
flowchart TD
    A["Every S1 × every S2/S3 record<br/>(millions × millions)<br/>❌ impossible to compare"] --> B["C9 blocking"]
    B --> C["24,104,327 candidate pairs<br/>for 49,996 S1 entities<br/>≈ 482 candidates each"]
    C --> D["159,970 of 173,213 true links<br/>are inside the shortlist<br/>= 92.35%"]
    D --> E["Matcher picks the real ones"]
```

Compared with the broad multi-key attempt (E2, about 4,615 candidates per entity), C9's shortlist is roughly **9.6× smaller** while recall rose from about 24% (E1) to **92.35%**.

---

## 🧹 Normalization

Memory-aware, done in `chunksize=100000` chunks so the full databases never sit in RAM.

```mermaid
flowchart LR
    A["Raw text<br/><i>Café & Co., Ltd.</i>"] --> B["lowercase"]
    B --> C["Unicode NFKD"]
    C --> D["strip accents"]
    D --> E["& → and"]
    E --> F["drop punctuation"]
    F --> G["collapse whitespace"]
    G --> H["<i>cafe and co ltd</i>"]
```

Both the **normalized name and normalized address** are kept, because either can be the deciding signal.

---

## 🧪 The 8-Experiment Journey to C9

Seven designs failed before one worked. The failures are kept on purpose: they explain why the final design looks the way it does.

```mermaid
flowchart LR
    E1["E1<br/>Prefix + address tail"] -->|"~24% recall"| E2["E2<br/>Broad multi-key"]
    E2 -->|"~4,615 cand/S1"| E3["E3<br/>Phonetic index"]
    E3 -->|"memory blow-up"| E4["E4<br/>Brute TF-IDF"]
    E4 -->|"too slow"| E5["E5<br/>Chunked TF-IDF"]
    E5 -->|"still slow"| E6["E6<br/>Char signatures"]
    E6 -->|"memory blow-up"| E7["E7<br/>Coarse blocker"]
    E7 -->|"0 new pairs"| E8["E8 = C9<br/>Multi-key + ranking"]
    E8 -->|"92.35% recall"| W["🏆 Adopted"]

    classDef bad fill:#f8d7da,stroke:#dc3545,color:#000;
    classDef good fill:#d4edda,stroke:#28a745,color:#000;
    class E1,E2,E3,E4,E5,E6,E7 bad;
    class E8,W good;
```

### 📋 Experiment Log

| # | Strategy | What happened | Verdict |
|:-:|---|---|:-:|
| E1 | Name prefix (4 chars) + address last token | **~24% recall**; too brittle for reordered names, typos, legal suffixes | ❌ |
| E2 | Broad multi-key (tokens, prefixes, Soundex, address combos) | **~4,615 candidates per S1**; search space unusable | ❌ |
| E3 | Phonetic (Soundex-style) indexing | Index expansion caused a **memory explosion** | ❌ |
| E4 | Brute-force character TF-IDF | Good representation, but nearest-neighbour over millions of rows was **too expensive** | ❌ |
| E5 | Chunked TF-IDF | Memory improved, compute **still too expensive** | ❌ |
| E6 | Generated character signatures | Huge Python index, **memory explosion** | ❌ |
| E7 | Coarse blocker (country + first/last char + length bucket) | **0 additional true pairs** | ❌ |
| **E8 / C9** | **Multi-key blocking ranked by shared-signal count** | **92.35% validated recall** | ✅ |

---

## 🔑 C9 Blocking Strategy

**Principle:** never depend on one fragile key. Use several independent keys and trust a pair more when **more keys agree**.

| Name keys | Meaning | | Address keys | Meaning |
|---|---|---|---|---|
| `NT` | Distinctive exact token | | `AT` | Exact address token |
| `NP5` | 5-char token prefix | | `AP6` | 6-char address prefix |
| `NP6` | 6-char token prefix | | `AN` | Number + address token |
| `NE` | Whole normalized name | | | |
| `NS` | Sorted normalized name tokens | | | |

### How a candidate gets chosen (illustrative example)

```mermaid
flowchart TD
    Q["S1: straight edge hypnosis"] --> K["Generate its keys"]
    K --> M1["NT: straight, edge, hypnosis"]
    K --> M2["NP5: strai, hypno"]
    K --> M3["NP6: straig, hypnos"]
    R["S2: straight edge hpyosris"] --> K2["Generate its keys"]
    K2 --> N1["NT: straight, edge, hpyosris"]
    K2 --> N2["NP5: strai, hpyos"]
    K2 --> N3["NP6: straig, hpyosr"]
    M1 --> S["Shared keys:<br/>straight, edge, strai, straig"]
    N1 --> S
    S --> SC["Score = 4 shared keys<br/>→ ranks high → kept ✅"]
```

Despite the typos in the third word, the pair still survives because other keys agree.

### Safeguards

```mermaid
flowchart LR
    A["Candidate key"] --> B{"Generic word?<br/>inc · llc · company · group …"}
    B -->|yes| X["Dropped ❌"]
    B -->|no| C{"Appears more than<br/>1200 times?"}
    C -->|yes| X
    C -->|no| D["Scope to country<br/>(country, key)"]
    D --> E["Add to index ✅"]
    E --> F["Rank candidates by<br/># shared keys"]
    F --> G["Keep top 500 per S1"]
```

```python
MAX_KEY_FREQUENCY     = 1200
MAX_CANDIDATES_PER_S1 = 500
MIN_TOKEN_LEN         = 3
```

---

## 📈 C9 Validation Results

Checked against the real ground-truth links for the exact S1 entities evaluated:

| Metric | Value |
|---|---:|
| S1 entities evaluated | 49,996 |
| Candidate pairs generated | 24,104,327 |
| Ground-truth true links | 173,213 |
| True links recovered | 159,970 |
| True links missed | 13,243 |
| **Blocking recall** | **92.35%** |

```mermaid
pie showData
    title True links (173,213)
    "Recovered by C9" : 159970
    "Missed by C9" : 13243
```

```
Recall = 159,970 / 173,213 = 92.35%
```

---

## 🔬 Missed-Pair Forensics

The 13,243 missed links (**9,606 unique S1 entities**) were pulled out and scored with RapidFuzz instead of guessed at.

| Similarity condition | Missed pairs | Share of missed |
|---|---:|---:|
| Name WRatio ≥ 70 | 9,799 | n/a |
| Address WRatio ≥ 70 | 9,809 | n/a |
| **Either field ≥ 70** | **13,064** | **≈ 98.65%** |
| **Either field ≥ 80** | **12,477** | **≈ 94.21%** |

```mermaid
flowchart TD
    A["13,243 missed true links"] --> B{"Either name or address<br/>similarity ≥ 70?"}
    B -->|"Yes: 13,064 (98.65%)"| C["Recoverable evidence exists ✅"]
    B -->|"No: 179 (1.35%)"| D["Truly hard cases"]
    C --> E["Strong name + weak address"]
    C --> F["Weak / missing name + strong address"]
    E --> G["💡 Matcher must use name + address +<br/>missingness + structure together"]
    F --> G
```

> 🧠 **Key finding:** the misses were not hopeless. Nearly all had usable evidence in the name **or** the address, so the matcher never relies on name similarity alone.

---

## 🤖 Matching Stage

```mermaid
flowchart LR
    A["C9 candidates"] --> B["Label from<br/>ground truth"]
    B --> C["Split by S1 entity"]
    C --> D["Build pair features"]
    D --> E["Mine hard negatives"]
    E --> F["Train GBM matcher"]
    F --> G["Entity-level<br/>threshold sweep"]
    G --> H["Final predictions"]
```

### Feature groups

```mermaid
mindmap
  root((Pair features))
    Name
      WRatio
      Ratio
      Token Sort / Set
      Levenshtein
      Jaccard
      Exact match
      Token overlap
      Length diff
    Address
      All name features
      Number overlap
      Postcode overlap
    Structural
      Country match
      Missing-name flag
      Missing-address flag
      Lengths
      Shared token counts
    Blocking
      Shared key count
      Which keys fired
      Candidate rank
      Candidate score
```

### Key design decisions

| Decision | Why |
|---|---|
| **Split by S1 entity**, never by row | Row-level splits leak candidates of the same S1 into both train and validation |
| **Hard negatives** (name sim 95%, address sim 90%, not a match) | Random negatives at 4% similarity teach nothing; near-duplicates teach "very similar ≠ same entity" |
| **Keep blocking signals as features** | Tells the model *why* a pair became a candidate, not just how similar it looks |
| **No default 0.5 threshold** | The metric is precision-heavy; sweep `0.50 … 0.95` and score per S1 entity (F0.5) |
| **Empty rows are valid** | A weak guess can score worse than correctly predicting no match |
| **No forced one-to-one** | Each S1 can match 0, 1 or many S2/S3 entities |

**Model:** LightGBM / XGBoost (fallback `HistGradientBoostingClassifier`) → outputs `P(match)`.

### Decision flow for each candidate pair

```mermaid
flowchart TD
    A["Candidate pair"] --> B["Compute features"]
    B --> C["Model → P(match)"]
    C --> D{"P ≥ locked threshold?"}
    D -->|yes| E["Add to this S1's matches"]
    D -->|no| F["Discard"]
    E --> G{"Any matches for this S1?"}
    F --> G
    G -->|yes| H["Write S1 → S2/S3 ids"]
    G -->|no| I["Write S1 with empty match ✅ valid"]
```

### Output format

```
S1-A    S2-123
S1-B
S1-C    S3-456
S1-D    S2-111,S3-222
```

### Final inference path

```mermaid
flowchart TD
    A["Full training data"] --> B["Train final matcher"]
    B --> C["C9 candidates for test S1"]
    C --> D["Feature engineering"]
    D --> E["Model probabilities"]
    E --> F["Locked threshold"]
    F --> G["matching_results.tsv"]
```

### Submission safety checks

```python
assert output_s1_count == test_s1_count
assert duplicate_s1_count == 0
assert invalid_entity_ids == 0
assert predictions_outside_candidate_set == 0
```

---

## ✅ Project Status: Complete

```mermaid
flowchart LR
    A["Exploration ✅"] --> B["Normalization ✅"] --> C["Blocking experiments ✅"] --> D["C9 candidates ✅"] --> E["Recall validation ✅"] --> F["Forensics ✅"] --> G["Features ✅"] --> H["Matcher ✅"] --> I["Threshold ✅"] --> J["Submission ✅"]
```

| Stage | Status |
|---|:---:|
| Dataset exploration | ✅ |
| Normalization | ✅ |
| Blocking experiments (E1 to E8) | ✅ |
| Candidate generation (C9) | ✅ |
| Candidate recall validation | ✅ |
| Missed-pair investigation | ✅ |
| Feature generation | ✅ |
| Matcher training | ✅ |
| Threshold optimization | ✅ |
| Test inference and submission | ✅ |

---

## 🧠 The One Idea to Remember

> **Don't make the blocker solve matching.**
>
> - The **blocker** asks: *"Could these two records plausibly be the same business?"*
> - The **matcher** asks: *"Given that they're plausible, are they actually the same business?"*

That split keeps the system **scalable, measurable, debuggable and model-friendly**.

---

## 🧪 Reproduce the Numbers

```python
# Recall
recovered = candidates.merge(truth, on=["s1_id", "match_id"]).shape[0]
recall = recovered / len(truth)            # 159,970 / 173,213 = 0.9235

# Missed-pair similarity (forensics)
from rapidfuzz import fuzz
missed["name_wr"] = [fuzz.WRatio(a, b) for a, b in zip(missed.s1_name, missed.m_name)]
missed["addr_wr"] = [fuzz.WRatio(a, b) for a, b in zip(missed.s1_addr, missed.m_addr)]
(missed[["name_wr", "addr_wr"]].max(axis=1) >= 70).mean()   # ≈ 0.9865
```

---

## 📁 Repository

```
Amazom_ML_Challenge_2026/
├── README.md
└── notebooks/        # exploration, candidate generation, validation, matching
```

The raw dataset is **not** included in this repository.

---

## 🙌 Credits

**Dharmi Sapariya** designed and built the solution: dataset exploration, the normalization approach, all eight blocking experiments, the C9 blocker, recall validation, missed-pair forensics, the matching-stage design, and the diagrams and documentation. The dataset was provided by Dharmi Sapariya (shared privately on Kaggle).

**Jasmine** ran the feature-generation step on the candidate pairs and supported the matching stage with model-training and threshold-tuning runs.

**Verify authorship** in the git history:

```bash
git shortlog -sn --all                 # commits per author
git log --stat --author="Dharmi"       # what each commit touched
git log --follow -p -- notebooks/      # full history of every notebook
```

Or see the [commit history](https://github.com/DharmiSapariya/Amazom_ML_Challenge_2026/commits/main/) and [contributors graph](https://github.com/DharmiSapariya/Amazom_ML_Challenge_2026/graphs/contributors).

<!-- Optional: add proof screenshots under docs/proof/ and embed them here:
![commit proof](docs/proof/git-shortlog.png) -->

---

Built for the Amazon ML Challenge 2026. This project is not affiliated with Amazon. Code released under the **MIT License**.
