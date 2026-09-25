Amazon ML Challenge 2026

Business Entity Resolution Pipeline

<p align="center">

High-Recall Candidate Generation → Hard-Negative Matching → Precision-Aware Entity Resolution

</p>
<p align="center">

Python · Pandas · RapidFuzz · scikit-learn · LightGBM/XGBoost · Kaggle

</p>

⸻

🧭 Project Status

Stage	Status
Dataset exploration	✅ Complete
Normalization	✅ Complete
Memory-aware preprocessing	✅ Complete
Blocking experiments	✅ Complete
Candidate generation	✅ Complete
Candidate recall validation	✅ Complete
Missed-pair investigation	✅ Complete
Matching features	🔄 Person B
Hard-negative training	🔄 Person B
Matcher	🔄 Person B
Threshold optimization	🔄 Person B
Full test inference	⏳
Final submission	⏳

Important: The 92.35% figure below is blocking recall on the validated 49,996-S1 evaluation sample, not the final competition score.

⸻

🎯 What Are We Solving?

The challenge is a noisy business entity resolution problem.

We have:

                 ┌──────────────────┐
                 │   S1 Reference   │
                 │  Deduplicated DB │
                 └────────┬─────────┘
                          │
                Find corresponding
                 businesses in
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │     S2      │         │     S3      │
       │ Noisy DB    │         │ Noisy DB    │
       └─────────────┘         └─────────────┘

For every S1 entity, the system must determine:

0 matches
1 match
multiple matches

The system must not assume one-to-one matching.

⸻

🧠 Why This Is Hard

The same business may appear as:

Primary Care Specialists Inc
Primary Care Specialists Incorporated
primary care stpianlsts inc

or:

Straight Edge Hypnosis
Straight Edge Hpyosris

Addresses can also contain:

* abbreviations
* missing components
* reordered components
* spelling differences
* transliteration
* landmarks
* numbering differences

Therefore:

Exact Match
       ❌
Normalization Only
       ❌
One Giant Fuzzy Search
       ❌
Two-Stage Entity Resolution
       ✅

⸻

🏗️ System Architecture

flowchart TD
A["Raw S1 / S2 / S3"]
B["Chunked Loading"]
C["Normalization"]
D["Multi-Key Blocking"]
E["Candidate Ranking"]
F["Candidate Pairs"]
G["Similarity Features"]
H["Hard Negatives"]
I["Matching Model"]
J["Probability"]
K["Entity-Level Threshold"]
L["Final Matching Results"]
A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
K --> L

The project is deliberately divided into two stages:

Stage A — Retrieval / Blocking

Find a high-recall set of plausible candidates.

Stage B — Matching

Decide which candidates are actually the same entity.

⸻

👨‍💻 Person A — Dharmi Sapariya

Data Engineering + Candidate Generation

My responsibility was the complete retrieval side of the entity-resolution pipeline.

This was not just preprocessing.

The main engineering problem was:

How do we reduce millions of possible comparisons while keeping true matches inside the candidate set?

A matcher cannot recover a true pair that blocking never gives it.

Therefore:

Candidate Recall
        ↓
Maximum possible Matcher Recall

This made candidate generation the first major optimization problem.

⸻

🔬 Work Completed

1. Dataset Exploration

The datasets were inspected for:

* schema
* missing values
* name quality
* address quality
* country distribution
* duplicates
* normalization requirements
* candidate explosion risks

The training ground truth was also used for proper candidate-recall validation.

⸻

2. Memory-Aware Normalization

The data contains millions of records, so preprocessing was designed around chunked loading.

Conceptually:

pd.read_csv(
    file,
    sep="\t",
    dtype=str,
    chunksize=100000,
    keep_default_na=False
)

Normalization included:

lowercase
   ↓
Unicode NFKD
   ↓
accent removal
   ↓
"&" → "and"
   ↓
punctuation removal
   ↓
non-alphanumeric cleanup
   ↓
whitespace normalization

Both normalized names and normalized addresses were retained.

⸻

3. Candidate Generation Was Iterative

Eight different approaches were investigated rather than assuming the first blocker was sufficient.

The goal of every experiment was measured against:

Recall
Candidate volume
Memory usage
Runtime
Scalability

⸻

🧪 Experiment 1 — Simple Prefix + Address Blocking

Initial idea:

Name first 4 characters
+
Address last token

Result

Approximately:

24% blocking recall

Why it failed

The representation was too brittle.

Business names can have:

different ordering
abbreviations
typos
legal suffix differences

and addresses can change substantially.

Decision

❌ Rejected.

⸻

🧪 Experiment 2 — Broad Multi-Key Blocking

Added multiple signals:

name tokens
name prefixes
Soundex
address tokens
address prefixes
digit/anchor combinations

with frequency limits and candidate caps.

This substantially increased coverage.

However:

~4,615 candidates / S1

on the evaluated sample created an extremely large downstream search space.

Decision

❌ Too broad for the main pipeline.

⸻

🧪 Experiment 3 — Phonetic Blocking

Phonetic representations such as Soundex were investigated for typo tolerance.

The problem was index expansion when many signatures and variants were combined.

Memory usage became impractical.

Decision

❌ Rejected as the primary index.

⸻

🧪 Experiment 4 — Character TF-IDF Retrieval

Character-level TF-IDF was investigated because it can tolerate:

typos
spacing differences
small edits
abbreviations

The first implementation attempted nearest-neighbor retrieval over millions of records.

The representation itself was useful, but brute-force nearest-neighbor computation became too expensive.

Decision

❌ Brute-force implementation rejected.

⸻

🧪 Experiment 5 — Chunked TF-IDF

TF-IDF retrieval was then investigated in chunks to reduce memory pressure.

Although chunking improved memory behavior, nearest-neighbor computation against the massive database was still too expensive.

Decision

❌ Not used as the production blocker.

⸻

🧪 Experiment 6 — Generated Character Signatures

A lightweight signature-based retrieval approach was investigated.

The problem was that combining:

phonetic signatures
deletions
character shapes
multiple variants

created a very large Python index.

Memory usage became unacceptable.

Decision

❌ Rejected.

⸻

🧪 Experiment 7 — Coarse Character Blocking

A strict blocker based on:

country
+
first characters
+
last characters
+
length bucket

was tested.

Validation showed:

Additional true candidate pairs recovered = 0

Decision

❌ Rejected.

⸻

🧪 Experiment 8 — C9 Multi-Key Blocking

The final validated approach combines multiple complementary blocking signals.

The principle:

Do not depend on one fragile key. Use several independent keys and rank candidates by how many signals support them.

⸻

🔑 C9 Blocking Strategy

Name keys

NT  → distinctive exact token
NP5 → 5-character token prefix
NP6 → 6-character token prefix
NE  → whole normalized name
NS  → sorted normalized name tokens

Address keys

AT  → exact address token
AP6 → 6-character address prefix
AN  → number + address token

Additional controls

country-aware keys
generic stopword filtering
maximum key frequency
maximum candidates per S1

⸻

🚫 Generic Token Filtering

Generic business terms were prevented from becoming dominant blocking signals.

Examples:

the
and
for
inc
llc
ltd
limited
company
corp
corporation
co
corporate
group
services
service
solutions
international
india
usa
us

This prevents a token such as:

"company"

from producing an enormous candidate group.

⸻

🌍 Country-Aware Blocking

Keys are effectively scoped as:

(country, blocking_key)

instead of simply:

blocking_key

This reduces unrelated cross-country candidate collisions while preserving the challenge’s open-set behavior.

⸻

📊 C9 Configuration

MAX_KEY_FREQUENCY = 1200
MAX_CANDIDATES_PER_S1 = 500
MIN_TOKEN_LEN = 3

Candidates are ranked using the number of independent blocking keys they share.

Conceptually:

candidate_score =
number of shared blocking signals

Then:

higher score
      ↓
stronger candidate
      ↓
candidate limit

⸻

📈 C9 Validation

The candidate generator was validated against the actual ground-truth relationships for the exact S1 entities being evaluated.

Validation sample

S1 entities evaluated       49,996
Candidate pairs             24,104,327
Ground-truth true links     173,213
True links recovered        159,970
True links missed            13,243

Blocking recall

159,970
──────────── × 100
173,213
= 92.35%

Current validated result

92.35% Blocking Recall

Again:

This is not the final competition score.

It is the measured recall ceiling of the validated candidate-generation pipeline on this evaluation sample.

⸻

🔍 Missed-Pair Forensics

The 13,243 missed true links were explicitly extracted and investigated.

13,243 missed pairs
        ↓
9,606 unique S1 entities

Instead of guessing why they were missed, normalized name and address similarity was calculated using RapidFuzz.

⸻

🧬 What the Missed Pairs Revealed

Name similarity

Name WRatio ≥ 70
9,799 pairs

Address similarity

Address WRatio ≥ 70
9,809 pairs

Either field ≥ 70

13,064 pairs
≈ 98.65%

Either field ≥ 80

12,477 pairs
≈ 94.21%

This was a major finding.

The remaining missed matches were often not completely dissimilar.

Many contained recoverable evidence in either:

business name

or:

business address

⸻

🧠 Critical Insight for Person B

Do not build a matcher that relies only on business-name similarity.

The missed-pair investigation demonstrated cases such as:

Strong name
+
weak address

and:

weak/missing name
+
strong address

The final feature model should therefore treat:

NAME
+
ADDRESS
+
MISSINGNESS
+
STRUCTURAL INFORMATION

as complementary evidence.

⸻

🤝 PERSON B — Matching Stage

Your Job Starts Here

The candidate generator is ready.

Do not restart candidate generation unless validation demonstrates a specific recall problem.

The next objective is:

C9 candidates
      ↓
Ground-truth labels
      ↓
Pair features
      ↓
Hard negatives
      ↓
Matcher
      ↓
Threshold tuning
      ↓
Final test predictions

⸻

1️⃣ Label the Candidate Pairs

For every C9 candidate pair:

match = 1

if the exact pair appears in:

train_ground_truth.tsv

Otherwise:

match = 0

Important:

Split by S1 entity.

Do not randomly split candidate rows.

Otherwise candidates belonging to the same S1 entity can leak information between train and validation.

Use:

S1 entities
    ↓
train S1
validation S1

rather than:

candidate rows
    ↓
random split

⸻

2️⃣ Build Strong Pair Features

For each candidate:

Name features

WRatio
Ratio
Token Sort Ratio
Token Set Ratio
Levenshtein similarity
Jaccard token similarity
exact normalized match
token overlap
length difference

Address features

WRatio
Ratio
Token Sort Ratio
Token Set Ratio
Levenshtein similarity
Jaccard similarity
exact normalized match
token overlap
number overlap
postcode overlap
length difference

Structural features

country match
name missing flag
address missing flag
name length
address length
shared token count
shared numeric tokens

Blocking features

Also preserve:

number of shared blocking keys
which blocking keys fired
candidate rank
candidate score

These features can tell the model not only:

“How similar are these records?”

but also:

“Why did these records become candidates?”

⸻

3️⃣ Hard Negative Mining

This is extremely important.

A random negative like:

similarity = 4%

teaches the model very little.

A hard negative like:

name similarity = 95%
address similarity = 90%
actual match = 0

is much more valuable.

Create negative samples from candidates with high similarity but incorrect ground-truth labels.

The model needs to learn:

Very similar
      ≠
Definitely the same entity

⸻

4️⃣ Train the Matcher

Start with a strong tabular model:

LightGBM

or:

XGBoost

Fallback:

HistGradientBoostingClassifier

The target:

1 = same entity
0 = different entity

The model outputs:

P(match)

⸻

5️⃣ Do NOT Automatically Use Threshold = 0.5

The challenge metric is precision-heavy.

The final threshold should be tuned against the actual validation objective.

Test a threshold grid such as:

0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95

But do not simply choose the threshold with the best pair-level accuracy.

Evaluate the actual entity-level behavior.

⸻

6️⃣ Evaluate Per S1 Entity

For every S1:

predicted matches
+
ground-truth matches

Calculate the challenge-style F0.5 behavior.

Pay special attention to:

false positives
false merges
missed true matches
correct empty predictions

⸻

🚨 Singleton / Empty Matches

Do NOT force every S1 to have a match.

Valid output:

S1-A    S2-123
S1-B
S1-C    S3-456
S1-D    S2-111,S3-222

The empty S1 row is valid.

A random weak match can be worse than correctly predicting no match.

⸻

7️⃣ Multiple Matches Are Allowed

Do NOT implement:

one S1 → exactly one S2/S3

The challenge explicitly allows:

one S1
   ↓
zero / one / multiple

Therefore the final inference stage should independently consider all sufficiently confident candidates.

⸻

8️⃣ Train → Validate → Error Analysis

Do not immediately train on everything.

Use:

TRAIN S1
   ↓
candidate labels
   ↓
features
   ↓
matcher
VALIDATION S1
   ↓
predictions
   ↓
threshold sweep
   ↓
entity-level evaluation

Then inspect:

False positives

Why did the model merge these businesses?

False negatives

Why did the model reject these true matches?

Difficult entities

blank name
+
strong address
strong name
+
blank address
common business name
multiple legitimate matches

Use these errors to improve features and thresholds.

⸻

🧪 Final Test Pipeline

Once validation is stable:

FULL TRAINING DATA
        ↓
TRAIN FINAL MATCHER
        ↓
GENERATE C9 CANDIDATES FOR TEST S1
        ↓
FEATURE ENGINEERING
        ↓
MODEL PROBABILITIES
        ↓
VALIDATED THRESHOLD
        ↓
FINAL MATCHES

⸻

📦 Required Final Files

The repository should eventually contain:

output/
├── matching_results.tsv
└── candidate_pairs.tsv

And:

code/
├── normalize.py
├── build_candidates.py
├── validate_candidates.py
├── build_features.py
├── train_matcher.py
└── generate_submission.py

Plus:

requirements.txt
README.md

⸻

✅ Final Output Validation

Before submission, automatically check:

Every test S1 appears

number of output S1 IDs
==
number of test S1 IDs

No duplicate S1 rows

duplicate S1 IDs = 0

No invalid entity IDs

Every predicted S2/S3 ID must actually exist.

Every predicted pair exists in candidate_pairs

predicted pair
      ↓
must exist in
candidate_pairs.tsv

Empty matches are allowed

Do not delete S1 entities with no confident match.

⸻

🧪 Recommended Final Validation Script

The final pipeline should automatically assert:

assert output_s1_count == test_s1_count
assert duplicate_s1_count == 0
assert invalid_entity_ids == 0
assert predictions_outside_candidate_set == 0

This prevents a technically good model from failing because of output-format mistakes.

⸻

📁 Recommended Repository Structure

Amazom_ML_Challenge_2026/
│
├── README.md
├── requirements.txt
│
├── code/
│   ├── normalize.py
│   ├── build_candidates.py
│   ├── validate_candidates.py
│   ├── build_features.py
│   ├── train_matcher.py
│   └── generate_submission.py
│
├── notebooks/
│   ├── person_a_candidate_generation.ipynb
│   ├── person_a_validation.ipynb
│   └── person_b_matching.ipynb
│
├── reports/
│   ├── candidate_recall.md
│   ├── error_analysis.md
│   └── experiment_log.md
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
└── artifacts/
    └── metrics/

Do not commit the massive raw datasets unless the competition explicitly permits and requires it.

Commit the code, notebooks, reports, configuration, and reasonably sized artifacts.

⸻

🧪 Experiment Log

Experiment	Strategy	Result	Decision
E1	Prefix + address tail	~24% recall	❌
E2	Broad multi-key	Very large candidate volume	❌
E3	Phonetic indexing	Memory explosion	❌
E4	Brute-force TF-IDF	Too expensive	❌
E5	Chunked TF-IDF	Still expensive	❌
E6	Generated signatures	Memory explosion	❌
E7	Coarse character blocker	0 additional pairs	❌
E8 / C9	Multi-key blocking	92.35% validated recall	✅

This experiment history is intentionally retained.

It shows why the current architecture exists.

⸻

📊 Current Candidate-Generation Snapshot

┌─────────────────────────────────────────┐
│          C9 VALIDATION                  │
├─────────────────────────────────────────┤
│ S1 evaluated             49,996         │
│ Candidate pairs          24,104,327     │
│ True links               173,213        │
│ Recovered                159,970        │
│ Missed                    13,243        │
│ Blocking recall          92.35%         │
└─────────────────────────────────────────┘

⸻

🧠 The Main Principle

Do not make the blocker solve matching.

The blocker should answer:

“Could these two records reasonably represent the same business?”

The matcher should answer:

“Given that they are plausible candidates, are they actually the same business?”

That separation keeps the system:

Scalable
+
Measurable
+
Debuggable
+
Model-friendly

⸻

🚀 Person B — Exact Execution Checklist

[ ] Load C9 candidate pairs
[ ] Label candidates from ground truth
[ ] Split by S1 entity
[ ] Build name similarity features
[ ] Build address similarity features
[ ] Build missingness features
[ ] Build numeric/postcode features
[ ] Preserve blocking features
[ ] Mine hard negatives
[ ] Train LightGBM/XGBoost
[ ] Validate at entity level
[ ] Sweep probability thresholds
[ ] Analyze false positives
[ ] Analyze false negatives
[ ] Lock threshold
[ ] Generate full-test C9 candidates
[ ] Generate test features
[ ] Run final matcher
[ ] Generate matching_results.tsv
[ ] Generate candidate_pairs.tsv
[ ] Validate output structure
[ ] Run final submission

⸻

🏁 Definition of Done

The project is complete only when:

                 ┌─────────────────────┐
                 │ Raw Challenge Data  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Normalization       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ C9 Candidate Gen    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Candidate Features  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Matching Model      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Threshold Tuning    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Test Predictions    │
                 └──────────┬──────────┘
                            ↓
              ┌───────────────────────────┐
              │ matching_results.tsv      │
              │ candidate_pairs.tsv       │
              └───────────────────────────┘

⸻

Team

Dharmi Sapariya

Data Engineering · Normalization · Blocking · Candidate Generation · Recall Validation · Error Analysis

Jasmine

Matching · Feature Engineering · Hard Negatives · Model Training · Threshold Optimization · Final Submission

⸻

<p align="center">

Amazon ML Challenge 2026

Entity Resolution under Real-World Noise

Normalize → Retrieve → Rank → Match → Validate

</p>
