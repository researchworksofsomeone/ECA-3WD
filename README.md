# ECA-3WD: certified three-way authentication of AI-generated images

Code for **ECA-3WD** (Evidence-Certified Asymmetric Three-Way Decisions). Given an image, the
detector does one of three things: **accept** it as real, **defer** it to review, or **flag** it as
generated. Two error rates are certified:

- **Accepted fakes.** The worst-case rate at which attacked fakes are accepted is bounded on
  average over generators never seen in calibration. The attack is any ℓ2 perturbation within the
  budget ε.
- **Framed reals.** The worst-case rate at which attacked real images are flagged is bounded.

The guarantee is an average over sources (generators). It is **not** an image-level certificate
of authenticity.

Everything (data, training, certification, calibration, attacks, analyses and paper figures) is
in one notebook, `eca3wd.py` (marimo), with a Jupyter export `eca3wd.ipynb`. No large language
models or agents are used anywhere in the pipeline.

> This repository is anonymised for review. It contains no author names, affiliations, account
> names, private repository identifiers or access tokens.

---

## Contents

| file | what |
|---|---|
| `eca3wd.py` | the notebook (marimo, 83 cells); the version to run |
| `eca3wd.ipynb` | the same notebook exported to Jupyter format, without outputs |
| `requirements.txt` | Python packages |
| `README.md` | this file |

---

## Method in brief

**Score.** A ConvNeXt-T base network `f` outputs the probability that an image is real. Gaussian
smoothing gives the realness probability `p(x) = P_η[f(x+η) ≥ 1/2]`, with `η ~ N(0, σ²I)`. The
score `S(x) = σ Φ⁻¹(p(x))` is 1-Lipschitz in ℓ2.

**Monte Carlo certification.** The notebook takes `N = 256` noise draws per image. Two-sided
Clopper–Pearson bounds at level `α_mc = 10⁻³` give a lower bound `S_lo` and an upper bound `S_hi`
on the score.

**Decision rule.** The rule uses three bands:

| decision | condition |
|---|---|
| accept | `S_lo ≥ τ_a + ε` |
| flag | `S_hi < τ_f − ε` |
| defer | otherwise |

The accept and flag regions are certified inner bounds of rough-set lower approximations under
the attacker's ℓ2 tolerance relation. An attack within budget can only move a certified decision
into the defer band.

**Calibration (CERT-CAL).** The two thresholds are set on held-out calibration data:

- **Flag threshold `τ_f`.** The conformal β-quantile of `S_lo` on calibration reals.
- **Accept threshold `τ_a = max(τ_CRC, τ_f)`.** `τ_CRC` comes from conformal risk control over
  generators: each generator is one exchangeable unit. Only generators with at least 20 images
  are used, on both the calibration and the test side.

**Training (ADRT).** Asymmetric decision-region training adds two hinge terms to the
noise-augmented cross-entropy:

- an accepted-fake term with weight `λ_AF`;
- a framing term with weight `λ_FR`.

`λ_FR` controls where `τ_f` falls.

**Guarantees.** Both bounds hold for any attack map chosen before the defender's fresh noise is
drawn:

- `E[AF_new generator(ε)] ≤ α + α_mc`
- `FR(ε) ≤ β + α_mc`

They hold conditionally on a model and a target `(α, β)` chosen on data independent of the final
calibration set.

---

## Requirements

- Python 3.10 or newer, with the packages in `requirements.txt` (`pip install -r requirements.txt`).
- One CUDA GPU for training and attacks. The runs used one 48 GB workstation GPU. Peak system RAM
  stays below 30 GB. The analysis cells (Part III A, Part IV) run on a CPU.

### Data sources (public, streamed from the Hugging Face Hub)

| dataset | role | used |
|---|---|---|
| `OwensLab/CommunityForensics-Small` | training, calibration and test (split by generator, 0.6/0.2/0.2) | 16 fake shards; reals FFHQ (3 shards), COCO (8), LandscapesHQ (4) |
| `ComplexDataLab/OpenFake` | out-of-population test (2025–26 generators) | first 3 shards, 200 images per generator, 2,400 reals |
| `aimagelab/RAID` | transfer adversarial examples | 400 ids per folder, ε ∈ {8, 16, 32}/255 |
| `stabilityai/sd-vae-ft-mse` | reconstruction pairs for the evidence-floor study | weights only |

Images are stored with a 256-pixel short side and centre-cropped to 224 × 224. Pixel values are
in [0, 1], so σ and ε are in pixel units.

---

## How to run

```bash
pip install -r requirements.txt
marimo edit eca3wd.py          # interactive
# or, headless:
RUN_STAGES=all ALLOW_TRAIN=1 python eca3wd.py
```

### Settings (environment variables only; nothing secret is written in the code)

| variable | default | effect |
|---|---|---|
| `HF_TOKEN` | unset | Hugging Face token. It is needed only to mirror results to a private repo. Set it as a notebook secret. |
| `HF_RESULTS_REPO` | unset | Private model repo (`<user>/<name>`) where results and weights are mirrored, so that a session that ends can resume. Without it, everything stays local. |
| `ALLOW_TRAIN` | `0` | `1` unlocks training in S5. Without it, S5 only fetches existing weights. |
| `RUN_STAGES` | empty | `all`, or a list such as `S1,S2,S5`. This runs Part I without the UI; otherwise pick stages in the RUN CONTROLS cell. |
| `ECA_WORK` | `eca_work` | Working folder. |
| `SMOKE` / `V3_SMOKE` | `0` | `1` uses tiny sizes, to test the code on a CPU. |
| `RAM_CAP_GB` / `VRAM_CAP_GB` | `30` / `70` | Memory guards. |

**Nothing finished is run twice.** Every stage checks for its output, locally or in the results
repo, before it does any work. Training resumes per epoch. GPU units in Parts III and IV skip any
unit whose output file exists. Folder names hash the configuration
(`data_<hash>`, `models_<hash>`, `results_<hash>`), and existing folders are reused when a session
resumes on another machine.

### Order of a full run

1. Part I: `S0` → `S1` → `S2` → `S3` → `S4` → `S5` (training, about 6 epochs per model) → `S6` → `S7` → `S8` (→ `S9`–`S12`).
2. Part II runs on its own once the S8 vote counts exist.
3. Part III: A1–A11 run in minutes on a CPU. B1–B5, C2 and C6 need a GPU and take hours, but resume.
4. Part IV: the figures, then the revision analyses R1 (CPU, minutes).

---

## Notebook map

### Part I: pipeline

| cell | what |
|---|---|
| 1–9 | setup, hardware report, config, Hub sync, memory plan, libraries (images and chunk store, models and datasets, smoothing / certification / calibration / metrics, attacks) |
| S0 | numerical self-tests of every theoretical claim (CPU) |
| S1 | index CommunityForensics-Small (metadata only); choose shards; split fakes by generator |
| S2 | stream the chosen shards, crop, and write a resumable chunk cache |
| S3 | OpenFake test subset (row-group-level fetch) |
| S4 | RAID transfer-attack set |
| S5 | train BASE, PGD-AT (ℓ∞ 2/255, 3 steps), Cohen (σ = 0.12) and ECA-3WD / ADRT (σ ∈ {0.06, 0.12, 0.25}); CLIP ViT-L/14 linear probe |
| S6 | uncertified baselines: scores; confidence, Mahalanobis and MC-dropout gates; conformal calibration |
| S7 | gate-blind and joint gate-aware attacks on the baselines |
| S8 | certification (vote counts), CERT-CAL, evaluation on CF / OpenFake / RAID / laundering |
| S9–S11 | adaptive EOT attacks and re-certification; evidence floor; sequential certification |
| S12 | report and coverage table |
| RUN STATUS / CONTROLS / RUNNER / VIEWER | stage selection and execution |

### Part II: corrected calibration

The framing guarantee needs `τ_f` calibrated on `S_lo` of real images. The first pass used
`S_hi`, so these cells correct it.

| cell | what |
|---|---|
| F1–F3 | resume and inventory; fill only missing evaluation units |
| F4 | re-calibrate every smoothed model from saved vote counts over the grid α, β ∈ {1, 2, 5, 10}% |
| F5a–F5e | S9–S11 again at the corrected thresholds; saliency check |
| F6–F7 | overall comparison tables and figures |

### Part III: analyses and new experiments

| cell | what |
|---|---|
| A1–A2 | pull and digest every saved result |
| A3 | feasibility frontier and certified operating curves |
| A4 | guarantee validity over 300 generator re-splits, and the calibration ablations |
| A5 | per-image certificate audit of all adaptive attacks |
| A6 | generator-cluster bootstrap intervals |
| A7 | three-way defended risk |
| A8 | OpenFake drift by release date |
| A11 | pooled vs generator-averaged error |
| A9 | research-question scoreboard |
| A10 | bundle zip |
| B1 | defended ROC of the baselines (GPU) |
| B2 | certified upper vs attacked lower bounds |
| B3 | attack stress test (GPU) |
| B4 | N = 1024 noise draws (GPU) |
| B5 | extra training seeds (GPU) |
| B6 | final audit |
| C1 / C1b | Yao's decision-theoretic three-way cost under attack, with the tight certified bound |
| C2 | stronger attack on PGD-AT (GPU) |
| C3 | AUROC by stratum |
| C4 | OpenFake adaptation |
| C5 | per-generator cap sensitivity |
| C6 / C6b–C6d | ADRT ablations and the λ_FR sweep, over seeds (GPU) |
| C7 | evidence-floor law |
| C8 | one-class collapse |
| C9 | manifest |

### Part IV: paper

| cell | what |
|---|---|
| FIG | every paper figure (PDF + 300-dpi PNG in `v3/paper_figs/`) and one zip of all outputs |
| R1 | frozen selection / final-calibration procedure with 200 re-splits; calibration ablations on the official split with attacked rates; flag/defer baseline and cost intervals; test strata; unbiased finite-N bound |

### Outputs

```
eca_work/
  data_<hash>/       crop caches, selection files
  models_<hash>/     weights (final.pt per model)
  results_<hash>/    stage results (json / csv / npz vote counts), comparison/
  v3/tables/         every analysis table (A*, B*, C*, R1_*.csv)
  v3/figs/, v3/paper_figs/
  eca3wd_analysis_bundle.zip, v3/eca3wd_all_artifacts.zip
```

---

## Results

### Data

- **CF calibration:** 248 generators (20–52 images each), 9,572 fakes and 8,941 reals.
- **CF test:** 249 generators and 12,243 fakes. Generator averages use the 248 generators with at
  least 20 images (12,225 fakes). There are 8,936 test reals.
- **Test generator sizes are heavy-tailed.** Three generators (CIPS, guidedDiffusion and
  StyleSANXL) hold 2,782 images; every other test generator has 18–51.

### Headline: frozen procedure

The model and the target are selected on one half of the calibration generators. The thresholds
are fitted on the other half, and the test set is used once.

| quantity | value (95% generator-cluster bootstrap) |
|---|---|
| selected model and target | ECA-0.06 (σ = 0.06), α = β = 5% |
| real acceptance at ε = σ = 0.06 | 92.2% [91.6, 92.7] (95.0% at ε = 0) |
| certified bound on attacked fake acceptance (event) | 2.3% [1.5, 3.5] |
| certified accepted-fake rate, clean test fakes | 1.5% [0.8, 2.5] |
| framing rate on test reals | 5.0% |
| 200 random selection / calibration / test splits | mean event/α = 0.50; event > α in 1 of 200; mean real acceptance 90.0%; models chosen: ECA-0.06 147, ECA-0.12 47, Cohen-0.12 6 |

**Reference protocol.** Here the thresholds are fitted on the full calibration set, so selection
reuses calibration data. ECA-0.12 at α = β = 5% and ε = σ = 0.12 gives:

- real acceptance 92.6%;
- event 4.3% [3.2, 5.7];
- accepted-fake rate 2.7% [1.8, 3.9];
- framing 4.8%.

### Certified operating points (CF test, α = β = 5%)

| model | accept ε=0 | AF gen. ε=0 | AF pooled ε=0 | accept ε=σ | AF gen. ε=σ | AF pooled ε=σ |
|---|---|---|---|---|---|---|
| Cohen-0.12 | 0.951 | 0.031 | 0.171 | 0.908 | 0.021 | 0.141 |
| ECA-0.06 | 0.950 | 0.020 | 0.157 | 0.923 | 0.015 | 0.137 |
| ECA-0.12 | 0.953 | 0.036 | 0.179 | 0.926 | 0.027 | 0.157 |
| ECA-0.25 | 0.910 | 0.045 | 0.180 | 0.860 | 0.032 | 0.153 |

Selected on calibration halves only, Cohen-0.12 accepted **no** real image at ε = σ. Its
threshold `τ_a/σ = 0.963` lies above the model-free cap `z_N − 1 = 0.892`. ECA-0.12 accepted 92.5%.

### Validity of the calibration (300 re-splits, ECA-0.12, target 5%)

| side | calibration | event | event / target | accept |
|---|---|---|---|---|
| accept | GEN-CRC on S_hi (ours) | 0.046 | 0.92 | 0.938 |
| accept | GEN-CRC on plug-in S | 0.078 | **1.56** | 0.947 |
| accept | GEN-CRC on S_lo | 0.496 | **9.93** | 0.955 |
| accept | image-level conformal on S_hi | 0.012 | 0.24 | 0.117 |
| flag | conformal on real S_lo (ours) | 0.050 | 0.99 | – |
| flag | conformal on real S_hi | 0.062 | **1.24** | – |

- GEN-CRC on S_hi met the target at every α ∈ {1, 2, 5, 10}% for all four smoothed models. The
  worst mean ratio was 0.95.
- **Attacked rates on the official split (ℓ2 ≤ 0.12).** For the accept-side rows, the observed
  certified acceptance of attacked fakes stayed below 5%: 3.0–4.2% for ours and 3.5–4.8% for the
  ablations. The ablations lose the certificate, not the observed error.
- **Flag side.** With `τ_f` calibrated on S_hi, attacked reals were certifiably flagged at
  6.0–6.4%, above β = 5%. Ours gave 2.0–3.2%.

### Per-image certificate audit

- **0 violations in 98 certified checks** over all pure-ℓ2 adaptive attacks, including a stress
  test. In 45 of these checks at least one attack reached its goal, so the checks are not vacuous.
- Laundering after the attack (blur, JPEG) is outside the ℓ2 certificate's scope: 27 of 200
  checks were violated there.

### Uncertified detectors under attack (ℓ2 = 0.25; joint / gate-blind attack)

| detector | confidence | Mahalanobis | MC dropout | all three gates |
|---|---|---|---|---|
| BASE | 0.68 / 0.68 | 0.66 / 0.61 | 0.34 / 0.24 | 0.27 / 0.20 |
| PGD-AT | 0.34 / 0.33 | 0.31 / 0.26 | 0.22 / 0.20 | 0.23 / 0.15 |
| CLIP probe | 0.94 / 0.94 | 0.84 / 0.46 | – | – |

- Gates calibrated to 1% accepted fakes reached 66–94% under gate-aware attacks.
- **ECA's certified upper bound was below the attacked lower bound of:**
  - CLIP up to 95% acceptance (ℓ2 ≤ 0.12);
  - BASE at ℓ2 = 0.25, up to 90% acceptance;
  - PGD-AT only at ℓ2 = 0.25, at 80–90% acceptance.
- **Adversarial training is empirically stronger at small budgets.** A 4× longer attack with 5
  restarts moved PGD-AT by at most 0.8 points at ℓ2 ≤ 0.12, so it is not gradient-masked.

### Three-way cost under attack

Loss (10, 0.5, 2, 0.2), priors ½ / ½. Certified rules report an upper bound, including the Monte
Carlo failure term 0.006. Uncertified rules report a lower bound under the observed attack, best
of BASE and PGD-AT.

| ℓ2 | ECA (certified) | Cohen (certified) | defer all | Yao rule | flag/defer, tuned under attack | flag/defer, tuned clean |
|---|---|---|---|---|---|---|
| 0.06 | 0.27 [0.21, 0.32] | 0.23 | 0.35 | 0.39 | 0.23 [0.19, 0.27] | 0.30 |
| 0.12 | 0.28 [0.21, 0.32] | 0.24 | 0.35 | 1.09 | 0.35 [0.35, 0.36] | 0.83 |
| 0.25 | 0.35 [0.30, 0.38] | 0.37 | 0.35 | 2.33 | 0.42 [0.39, 0.45] | 1.12 |

- At ℓ2 = 0.06 a flag/defer rule tuned under the attack **matches** the certified bound. At this
  budget the advantage over Yao comes from the action policy.
- At 0.12 and 0.25 the certified bound is lower, and the intervals do not overlap.
- The cost-optimal certified policy never accepts: 68–71% of attacked fakes are certifiably
  flagged.
- Cohen's certified cost is slightly lower than ECA's (λ_FR = 1) at ℓ2 ≤ 0.12.

### Accept/flag trade-off law and the ADRT knob

**Model-free window.** At N = 256 and ε = σ, certified acceptance is impossible once
`τ_f/σ > z_N − 1 = 0.892`, whatever the network. All 15 trained models obey this law, including
two seeds whose outcome was predicted before they were trained (`τ_f/σ` = 0.91 and 1.00, and both
accepted nothing).

The table reports σ = 0.12 on balanced subsets, mean ± s.d. over 3 seeds where shown. Cost uses
the loss above at ε = σ/2.

| variant | τ_f/σ | α_min(σ) | accept | event | AUROC | certified flag at ε=σ/2 | cost at ε=σ/2 |
|---|---|---|---|---|---|---|---|
| λ_FR = 0 (fake term only) | −1.42 | 2.06 ± 0.21% | 0.927 | 0.048 | 0.977 | 0.00 | 0.54 |
| λ_FR = 1 | −0.48 ± 0.23 | 2.40 ± 0.13% | 0.925 | 0.041 | 0.9820 | 0.58 ± 0.49 | 0.36 ± 0.14 |
| **λ_FR = 4** | +0.49 ± 0.16 | 2.72 ± 0.19% | 0.913 | **0.035** | **0.9837** | **0.952 ± 0.003** | **0.27 ± 0.05** |
| λ_FR = 8 | +0.90 ± 0.11 | 3.09 ± 0.12% | 0.894 / 0 / 0 | 0.035 | 0.9827 | 0.953 | 0.33 |
| real term only | +1.14 | 3.49% | 0.000 | 0.036 | 0.982 | 0.95 | 0.33 |
| Cohen (no ADRT) | +0.84 | 3.33% | 0.906 | 0.036 | 0.9812 | 0.95 | 0.33 |

**Seed stability (ECA-0.12, 3 seeds):**

- real acceptance at ε = σ: 0.930, 0.922 and 0.923;
- accepted-fake rate: 0.027, 0.023 and 0.028;
- AUROC: 0.982, 0.983 and 0.982.

### How low can α go? (finite-N confident-error bound)

| N | m̂_N (fakes with every vote real) | bound on α_min | measured α_min(σ) |
|---|---|---|---|
| 64 | 0.942% | 1.34% | – |
| 128 | 0.857% | 1.26% | – |
| 256 | 0.786% | 1.18% | 2.27% |
| 512 | 0.722% | 1.12% | – |
| 1024 | 0.665% | 1.06% | 1.81% |

- α = 1% is infeasible at every measured N ≤ 1024.
- Raising N from 256 to 1024 raised real acceptance at α = 2% from 0% to 85.1%.
- The provable limit as N → ∞ is only 1/(K+1) = 0.40%. Extrapolation beyond N = 1024 is
  heuristic.

### Generator-level vs image-level error (source mass)

- **ECA-0.12:** 2.7% generator-averaged against 15.7% pooled, and 4.6% with each generator
  counted at most 100 times.
- **Three generators carry 90% of accepted fakes:** CIPS (821 of 891), guidedDiffusion
  (481 of 1,036) and StyleSANXL (426 of 855). Their pooled rate is 62%, against 2.1% on the other
  245.
- The gap equals `Cov_g(w_g, r_g)` exactly (source-mass identity).

### Transfer to a new family (OpenFake)

- **The guarantee does not transfer.** The event is 0.137–0.236 at α = 5%, and 80–100% of the 20
  OpenFake generators exceed α. There is no significant trend with release date (Spearman ρ
  between −0.36 and −0.15, p ≥ 0.22).
- **One-class collapse.** For α ≥ α* (4.0% for ECA-0.12 at β = 5%), `τ_a = τ_f`. The accept rule
  then made identical decisions to a one-class test on reals for 100% of images.

### AUROC and efficiency

- **AUROC.** Smoothing lowers AUROC by 5.1–7.2 points (BASE 0.977; ECA 0.906–0.926). On the 245
  generators with at most 100 images, ECA-0.12 loses only 1.4 points. On the three large
  generators it loses 20.
- **Sequential certification.** It needs 96, 121 and 118 noise draws per decision for the three
  ECA models, against 261 for Cohen. Its decisions agree with fixed-N certification for 98–100%
  of images, against 53% for Cohen.

---

## Limitations

- The certificate covers ℓ2 perturbations only. An ℓ∞ ball of radius 1/255 reaches ℓ2 distance
  1.52 in 224 × 224 × 3, far beyond the largest certifiable radius (0.227 at σ = 0.12).
  ℓ∞, patch, semantic and laundering attacks are outside the guarantee.
- **The guarantee assumes test generators are exchangeable with calibration generators.** A new
  family (OpenFake) breaks this.
- **One backbone, one training budget and one benchmark for calibration.** Seeds and loss
  variants are studied on balanced subsets.
- **The frozen headline model certifies a budget of 0.06.** That is half the reference model's
  budget.

---

## Reproducibility

- **Seeds.** The global seed is 0. Splits and subsets are chosen by deterministic hashes, never
  by random state.
- **Folder naming.** Every stage's output folder name hashes the configuration that produced it.
- **Confidence intervals.** These are generator-cluster bootstrap intervals (B = 1000–2000) or
  Clopper–Pearson intervals.
- **No selection on test data.** All thresholds, targets and model choices are made on
  calibration data.

## Citation

Withheld for anonymous review.
