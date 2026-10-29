# Signatur: methodology

How the geophysical habitat-potential model is built, validated and turned into
maps. It covers every step of the production pipeline, from the habitat
polygons to the plausible-habitat sets and habitat groups.

- **Production model:** the climate-inclusive model (`config_climate.yaml`;
  outputs in `output_climate/`).
- **Status:** all numbers below were checked against those outputs on
  29 October 2026.
- **Related documents:**
  - [plausible_class_sets.md](plausible_class_sets.md): a longer, non-technical
    explanation of the plausible sets.
  - [review_report_2.7-2.10.md](review_report_2.7-2.10.md): comments on the
    report text.

Notation follows the report (sections 2.7–2.10): $D_k^2$, $S_k$, $L_k$,
$p(k\mid x)$, $q_k$, $r_k$, $\mathcal{H}(x)$ and $P(g\mid x)$.

---

## 1. Data

### 1.1 Habitat reference data

Three Danish habitat-polygon layers, each with a Natura 2000 habitat code
(`naturkode`):

| Layer | Polygons | Priority |
|---|---|---|
| `bevaringsvrdige_egekrat` (oak scrub of conservation value) | 878 | 1 (highest) |
| `np3__lysbne_naturtyper` (open habitat types) | 19,908 | 2 |
| `np3__skovnaturtyper` (forest habitat types) | 8,145 | 3 |
| **Total** | **28,931** | |

**Rasterisation and overlaps.** The polygons are rasterised onto the national
10 m grid (EPSG:25832), where a cell belongs to a polygon if its centre falls
inside it. Where layers overlap, the higher-priority layer wins:

| Layer | Cells lost to a higher-priority layer |
|---|---|
| open habitat types | 5,911 |
| forest habitat types | 122,387 |

**Result:** 13,504,989 habitat cells (135,050 ha) in 45 habitat classes.
Habitat codes are mapped to integer class IDs through a lookup table
(`00_lookup/`), because some codes are alphanumeric (e.g. 91D0, 91E0).

### 1.2 Geophysical predictors

There are 11 principal components (PCs) from the GeoILUM geo-ecological sphere
model, all on the same 10 m grid:

| Sphere | Components |
|---|---|
| Hydrosphere | `hydro_PC1`, `hydro_PC2` |
| Soil | `soil_PC1`–`soil_PC3` |
| Toposphere | `toposphere_PC1`–`toposphere_PC3` |
| Climate | `climate_PC1`–`climate_PC3` |

The climate PCs are originally at 1 km. They were resampled onto the 10 m grid
by bilinear interpolation (`climateprep`). Bilinear was chosen over nearest
neighbour so that the smooth climate surface does not get artificial 1 km block
edges.

- **Grid:** 45,151 × 35,251 cells, of which about 403.3 million are Danish land
  with valid values for all predictors.
- **Why share one grid:** every habitat cell and every map cell is the same cell
  in all rasters. Training extraction and national projection therefore read the
  rasters block by block, with no coordinate lookup.

### 1.3 Train/test split (holdout)

The split is made at **polygon level**, not cell level:
- within each habitat class, each polygon is independently assigned to the test
  set with probability 0.2 (seeded);
- the split is stored per polygon (`01_merged_raster/feature_split.csv`) and
  carried to every cell.

| | Polygons | Cells |
|---|---|---|
| Train (80 %) | 80.1 % | 10,738,759 |
| Test / held-out (20 %) | 19.9 % | 2,766,230 |

Cells within one polygon are spatially autocorrelated, with near-identical
geophysics. A cell-level split would put near-copies of the same site in both
training and test data and inflate every accuracy measure. Splitting whole
spatial units is standard practice for spatially structured data (Roberts et al.
2017).

---

## 2. The habitat envelope model

### 2.1 Standardisation

Each predictor is:
1. **winsorised** at the training distribution's mean ± 5 standard deviations,
   then
2. **z-scored** with the mean and standard deviation of the winsorised values.

This limits the leverage of extreme values before the distance-based modelling.
The scaling parameters are stored (`05_model/scaler.json`) and reused unchanged
in the national projection.

### 2.2 One Gaussian envelope per habitat class

Each habitat class $k$ with at least 150 training cells is modelled as one
multivariate Gaussian in the standardised 11-dimensional space:
- a mean vector $\mu_k$;
- a covariance matrix $\Sigma_k$, estimated with **Ledoit–Wolf shrinkage**
  (Ledoit & Wolf 2004; `sklearn.covariance.LedoitWolf`).

Shrinkage gives a well-conditioned, invertible covariance estimate even when
predictors are correlated and a class has moderate sample size.

44 of the 45 classes get an envelope. Class 6430 has only 16 cells and is
excluded.

**Two fits of the same model are used:**

| Fit | Data | Used for |
|---|---|---|
| Production envelopes (`envelopes.npz`) | all polygons | every national map |
| Calibration envelopes (`envelopes_train.npz`) | the 80 % training polygons only | validation (§5) and calibration of the plausible sets (§7–8) |

### 2.3 Distance and suitability

The squared Mahalanobis distance from a location with predictor vector $x$ to
class $k$ (Mahalanobis 1936) is:

$$D_k^2(x) = (x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k)$$

It is converted to a 0–1 suitability with the survival function of the
χ² distribution with 11 degrees of freedom:

$$S_k(x) = 1 - F_{\chi^2_{11}}\!\left(D_k^2(x)\right)$$

How to read $S_k$:
- **1:** at the centre of the class's occupied niche.
- **About 0.5:** as typical as the median occupied cell.
- **Near 0:** far outside.

$S_k$ is a measure of **environmental compatibility** (absolute: "is this cell
typical of class $k$?"). It is **not** a probability that the habitat will
establish.

**How well the Gaussian fits.** In the training data, the share of a class's own
cells inside its χ² 95 % shell is a median of 0.913 (lowest 0.865), against a
target of 0.95 (`05_model/model_report.csv`). So the Gaussian is a reasonable
approximation within the training polygons. **At new sites the fit is worse:**
9.5 % of held-out cells lie outside their own class's 99 % Gaussian envelope,
against an expected 1 %. Variation *between* sites is larger than the variation
*within* the training polygons. This is why the envelope boundary used in the
plausible sets is calibrated empirically (§7.3).

### 2.4 National projection

The production envelopes are projected over the whole grid, block by block
(`project`, outputs in `06_fuzzy/`):

| Output | Content |
|---|---|
| `NN_<code>_suitability.tif` | $S_k$ for each of the 44 classes |
| `best_class.tif` | the class with the smallest $D_k^2$, which is the same as the largest $S_k$ |
| `best_suitability.tif`, `best_distance.tif` | $\max_k S_k$ and the corresponding Mahalanobis distance |

`best_class` is a **distance-only** winner. It is not the same as the
most-probable class in §6. See the note there.

---

## 3. Habitat richness

For every cell (`richness`, outputs in `07_potential/`):

| Output | Content |
|---|---|
| `habitat_richness.tif` | the number of classes with $S_k \ge 0.5$ |
| `habitat_richness_open.tif` | the same count for open, non-forest habitats only (main types Hede, Overdrev, Mose, Klit, Kyst, Klippe) |
| `habitat_richness_sum.tif` | $\sum_k S_k$, a threshold-free "effective number of habitats" |

Richness is a type-agnostic indicator of how many habitat types the geophysics
could support. It is useful for delineating restoration areas.

---

## 4. Residual analysis: land outside every envelope

**Categories.** `best_suitability` is grouped into χ² confidence bands
(`09_residual/residual_category.tif`):

| Category | Rule | Area |
|---|---|---|
| core | $\max_k S_k \ge 0.5$ | 3,483,408 ha |
| marginal | $0.05 \le \max_k S_k < 0.5$ | 490,318 ha |
| outside 95 % | $0.01 \le \max_k S_k < 0.05$ | 27,468 ha |
| outside 99 % | $\max_k S_k < 0.01$ | 31,574 ha |

**Clustering.** The roughly 59,000 ha outside the 95 % band were sampled
(300,000 cells) and clustered with k-means in the standardised PC space. The
number of clusters (k = 6) was chosen by silhouette score: 0.352, the best for
k = 3–10. Each cluster is profiled by its PC means and its nearest habitat
envelope (`cluster_profile.csv`, radar charts):

| Cluster | Area | Nearest habitat | $S$ to it | Profile |
|---|---|---|---|---|
| 4 | 14,364 ha | 6210 | ≈ 0 | extreme soil_PC2/PC3 (the soil-chemistry signature of raised bogs), but topography and hydrology match no bog envelope |
| 1 | 13,649 ha | 1230 | ≈ 0 | extreme toposphere_PC1/PC2, colder climate |
| 0 | 12,538 ha | 6210 | 0.27 | low soil_PC1 (calcium-rich), otherwise unremarkable |
| 3 | 11,340 ha | 5130 | 0.26 | very high hydro_PC2 |
| 5 | 6,955 ha | 9190 | 0.49 | elevated hydro_PC1, dry, nutrient-poor soil |
| 2 | 156 ha | 1230 | ≈ 0 | extreme toposphere on both axes |

**Interpretation (hypotheses, not verified against land-use data).**
- Clusters 1, 2 and 4 (about 28,000 ha) resemble no surveyed habitat. They are
  probably artificial or heavily modified land. Cluster 4 may be drained peat:
  peat chemistry without a functioning bog.
- Clusters 0, 3 and 5 (about 31,000 ha) are near-misses just outside real
  envelopes.

Both readings should be checked against land use and field knowledge before
being used.

---

## 5. Validation

### 5.1 In-sample separability check (not used to build the maps)

- **Method:** a stratified 5-fold cross-validated RandomForest on the 11 PCs,
  using a class-stratified sample of 120,000 habitat cells and the 44 classes
  with at least 50 cells (`04_stats/`).
- **Result:** balanced accuracy **0.71**, i.e. 31.4 × chance with 44 classes.
  Permutation importance of each PC is reported alongside.
- **Caveat:** because folds are split at cell level, cells of the same polygon
  fall in both training and test folds. This number is therefore **optimistic**.
  It shows only that the classes occupy distinguishable regions of the predictor
  space.

### 5.2 Holdout validation (the honest numbers)

Both models are fitted on the 80 % training polygons and evaluated only on the
20 % held-out polygons (`08_holdout/holdout_report.md`):

| Model | Top-1 accuracy | Balanced accuracy | × chance |
|---|---|---|---|
| Envelope model (nearest envelope; drives the maps) | 0.265 | **0.260** | 11.7 |
| RandomForest (separability cross-check) | 0.467 | **0.382** | 17.2 |

- **How suitable the true class looks:** the median suitability $S_k$ of the
  *true* class at held-out cells is 0.756.
- **Test set details:** 45 classes appear in the test set, so chance is
  1/45 ≈ 0.022. Eleven test cells of the class without an envelope (6430) are
  excluded from the envelope evaluation.
- **Comparison with §5.1:** the large drop from 0.71 to 0.38 for the
  RandomForest shows how much cell-level leakage inflated the in-sample figure.

### 5.3 The "× chance" measure

For an $n$-class problem, a classifier that guesses uniformly at random has an
expected balanced accuracy of exactly $1/n$. Balanced accuracy is the mean of
the per-class recalls, and each class gets an expected recall of $1/n$ under
uniform guessing, however imbalanced the classes are. Hence:

$$\text{multiplier} = \text{balanced accuracy} \times n$$

### 5.4 Over-predicted classes

From the holdout confusion matrix, the *precision* of each class is the share
of cells predicted as that class that truly belong to it
(`08_holdout/holdout_confusion_precision.csv`). Classes with low precision and
many predicted cells have broad, "greedy" envelopes:

| Code | Habitat | Precision | Cells taken from other true classes |
|---|---|---|---|
| 2110 | Forklitter | 0.023 | 245,174 |
| 2120 | Hvide klitter | 0.064 | 187,972 |
| 2130 | Grå/grøn klit | 0.224 | 171,218 |
| 9160 | Ege-blandskov | 0.113 | 140,694 |
| 2190 | Klitlavning | 0.302 | 132,125 |
| 1230 | Kystklint | 0.036 | 103,096 |
| 6410 | Tidvis våd eng | 0.030 | 94,696 |
| 2180 | Skovklit | 0.012 | 46,898 |

Whether this reflects real over-estimation or a genuinely broad geophysical
niche is a question for domain judgement.

---

## 6. Relative class probabilities and the most compatible class

Suitability $S_k$ answers "is the cell typical of $k$?" separately for each
class. To compare classes, the Gaussian likelihoods are normalised across the
44 classes, with **equal priors**:

$$L_k(x) \propto |\Sigma_k|^{-1/2}\exp\!\left(-\tfrac12 D_k^2(x)\right),\qquad
p(k\mid x) = \frac{L_k(x)}{\sum_{j=1}^{44} L_j(x)}$$

- **Why equal priors:** today's area of each habitat does not influence the
  potential maps.
- **$p(k\mid x)$ is relative:** it depends on which classes are in the model,
  and it sums to 1 over the 44 classes even where the cell is a poor match to
  all of them.

**Most compatible class** (`10_sets/map_class.tif`): the class with the highest
$p(k\mid x)$ **among the classes that pass the envelope condition** (§7.3),
falling back to the overall highest where no class does. Its probability is
stored in `best_posterior.tif`, as a measure of how clearly it wins.

**Difference from `best_class.tif`** (§2.4). The likelihood includes the term
$|\Sigma_k|^{-1/2}$:
- a narrow (specialised) envelope counts more strongly than a broad one at the
  same distance;
- `best_class` uses distance only, and so favours broad envelopes.

The two maps can therefore differ. The report should say which one a figure
shows.

---

## 7. Plausible habitat sets (conformal prediction)

`setcalib` and `sets`; outputs in `08_holdout/set_*` and `10_sets/`. A more
detailed explanation is in [plausible_class_sets.md](plausible_class_sets.md).

### 7.1 Why

A single best class hides near-ties: a cell where the runner-up is almost as
good looks exactly like a cell with one clear winner. Instead, each cell gets
the **set of classes that cannot be ruled out**. Its size adapts to the cell,
and it is calibrated so that its meaning can be tested on unseen data.

### 7.2 Definition

$$\mathcal{H}(x) = \{\,k : p(k\mid x) \ge q_k \;\wedge\; D_k^2(x) \le r_k\,\}
\;\cup\; \{k^{*}_{\text{adm}}(x)\}$$

- **$q_k$:** the calibrated probability threshold for class $k$ (§7.4).
- **$r_k$:** the calibrated envelope radius for class $k$ (§7.3).
- **$k^{*}_{\text{adm}}(x)$:** the highest-probability class among those with
  $D_k^2(x) \le r_k$. It is always included, so the set is non-empty whenever at
  least one envelope contains the cell.
- **Empty set:** a cell inside no calibrated envelope gets $\mathcal{H}(x) = \varnothing$.

### 7.3 Envelope condition with a calibrated radius

**Why there is an envelope condition at all.** The probability is relative, so
a class can "win" in a cell outside its own niche simply because all other
classes are even further away. The envelope condition admits a class only where
the cell lies inside that class's own envelope.

**Why the radius is calibrated.** The Gaussian χ² radius proved far too narrow
for new sites (§2.3: 9.5 % of held-out cells outside their own 99 % envelope).
Using it excluded the true class up front at about 10 % of sites, and inflated
the sets to 22.7 classes on average. The radius is therefore calibrated
empirically:

- **Definition:** $r_k$ is the $D_k^2$ that contains **98 %** of class $k$'s
  held-out cells.
  - Each held-out polygon gets equal total weight (cells weighted 1 / cells in
    the polygon).
  - The quantile uses the finite-sample correction (§7.4).
  - Classes with fewer than 10 held-out polygons get a pooled radius.
- **Size:** the calibrated radii are a median of 2.7 times the Gaussian 98 %
  radius in $D^2$, and much larger for some classes.
- **Why 98 % and not 99 %:** at 99 % the radius of several classes was set by one
  or two extreme polygons. For 6210 it became 15,915 times the Gaussian radius,
  which in practice switches the condition off.

### 7.4 Thresholds (Mondrian split-conformal calibration)

The thresholds $q_k$ are calibrated on the **held-out polygons**, with
probabilities computed from the **calibration envelopes** (fitted without those
polygons):

1. **Score per cell.** For every held-out cell of true class $k$, the score is
   $-\log p(k\mid x)$, set to $+\infty$ if $D_k^2(x) > r_k$.
2. **Quantile.** $q_k$ is set from the polygon-weighted quantile of these scores
   at level $\min\{1,\,0.9\,(1 + 1/n_k)\}$, where $n_k$ is the number of
   held-out polygons of class $k$. That is the standard finite-sample correction
   of split conformal prediction.
3. **Pooled threshold.** Classes with fewer than 10 held-out polygons
   (7 of 44) use a threshold calibrated over all held-out cells.

**What the guarantee means.** Calibrating per class (the "Mondrian" form) gives
about 90 % coverage **for each class**, not just on average over classes.
- It is *marginal*: across many new sites of class $k$, about 90 % have $k$ in
  their set. It is not a 90 % probability for each individual cell.
- It is measured on existing habitat sites. Applied to farmland or built-up land,
  the sets say which habitats the conditions are compatible with, at the
  precision achievable at real habitat sites. That is an extrapolation.

### 7.5 Honest evaluation (cross-conformal)

Evaluating thresholds on the polygons used to calibrate them would be circular.
So:
1. The held-out polygons are split into two halves, stratified by class and
   seeded.
2. Radii **and** thresholds are fitted on one half and evaluated on the other,
   in both directions.

The final $r_k$ and $q_k$ used for the maps are fitted on all held-out polygons.

### 7.6 Results

Cross-validated, on held-out polygons (`08_holdout/set_report.md`):

| Rule | True class in the set | Mean set size |
|---|---|---|
| **Conformal, 90 %** | **0.905** | **14.1** of 44 |
| Likelihood ratio (within 1/8 of the best) | 0.46 | 2.8 |
| Most compatible class only | 0.22 | 1 |

- **Envelope condition:** it rules out the true class at 1.5 % of held-out
  cells, and 0.1 % of held-out cells lie inside no envelope.
- **Classes below 85 % coverage:** 1310, 1320, 1340, 2250, 2330, 7150, 9170. All
  have few held-out polygons.
- **Nationally** (`10_sets/set_size_summary.csv`):
  - mean set size 15.7 habitats;
  - only 2.0 % of the land has 1–4 plausible habitats;
  - 0.01 % of the land is inside no calibrated envelope.

**Reading.** The guarantee holds, but the sets are large: the 11 PCs separate
*groups* of habitats well, but often cannot separate individual types within a
group. This is consistent with the holdout top-1 accuracy of about 0.26. It is a
property of the data, not a flaw in the method. A "top-3" map would look precise
while missing the true habitat at more than half of the sites.

---

## 8. Habitat groups (two-level sets)

`groupcalib` and `groups`; outputs in `08_holdout/group_*` and `11_groups/`.

### 8.1 Groups

The 44 classes are grouped into seven groups (`data/types and classes.csv`,
column `Klasse nr`):

| No. | Group | Habitats |
|---|---|---|
| 1 | Beaches and salt marshes (Strande og strandenge) | 1210, 1220, 1230, 1310, 1320, 1330, 1340 |
| 2 | Coastal dunes (Kystklitter) | 2110, 2120, 2130, 2140, 2160, 2170, 2190, 2250 |
| 3 | Inland dunes (Indlandsklitter) | 2310, 2320, 2330 |
| 4 | Heath and scrub (Heder og krat) | 4010, 4030, 5130 |
| 5 | Grasslands, meadows and rock (Overdrev, eng og klipper) | 6120, 6210, 6230, 6410, 8220 |
| 6 | Mires (Moser) | 7110, 7120, 7140, 7150, 7210, 7220, 7230 |
| 7 | Forests (Skove) | 2180, 9110, 9120, 9130, 9150, 9160, 9170, 9180, 9190, 91D0, 91E0 |

The grouping comes from the file, not from the first two digits of the code.
Wooded dunes (2180) are deliberately placed under forests.

### 8.2 Group sets

The group probability is the sum of the class probabilities:

$$P(g\mid x) = \sum_{k\in g} p(k\mid x)$$

The group's likelihood is thus a mixture of its classes' envelopes, not one
large envelope.
- **Threshold:** group-specific thresholds are calibrated as in §7.4, at
  **80 %** coverage of the true group.
- **Envelope condition:** a group is admissible only if at least one of its
  classes passes its envelope condition ($D_k^2 \le r_k$, same radii as §7.3).
  Without this condition, summing could make a group look certain in the gap
  between its classes' niches.
- **Always included:** the highest-probability admissible group.

### 8.3 Habitats within a plausible group

The conditional probability within group $g$ is:

$$p(k\mid g, x) = \frac{p(k\mid x)}{P(g\mid x)}$$

- **Threshold:** thresholds are calibrated on held-out cells *of the true
  group*, at **90 %** coverage.
- **Which classes are listed:** a class is listed only if its group is plausible
  and the class passes its own envelope condition.
- **Always included:** the best admissible class of every plausible group.

### 8.4 Results

Cross-validated, on held-out polygons (`08_holdout/group_report.md`):

| Level | True one in the set | Mean set size |
|---|---|---|
| **Group (80 %)** | **0.801** | **2.21** of 7 |
| Class within its true group (90 %) | 0.903 | 3.8 |
| Best group only | 0.558 | 1 |

- **Held-out sites:**
  - group set sizes: 1 group 28 %, 2 groups 35 %, 3 groups 27 %, 4 groups 10 %,
    5 groups 1 %;
  - the envelope condition rules out the true group at 0.3 % of sites.
- **Nationally** (`11_groups/group_combinations.csv`):
  - 120 distinct group combinations occur (of 127 possible);
  - the 10 largest cover 70.7 % of the land, the 20 largest 86.2 %;
  - group set sizes: 1 group 21.9 %, 2 groups 40.6 %, 3 groups 28.2 %, 4 groups
    8.7 %, 5 groups 0.7 %;
  - the largest combinations are: grasslands + forests (21.1 %), forests only
    (13.2 %), heath + grasslands + forests (9.9 %), grasslands + mires + forests
    (7.0 %), inland dunes + heath (6.1 %).

**Combined coverage.** A site's true *habitat* appears in the two-level output
only if its group is in the group set (≈ 80 %) **and** the habitat passes the
within-group threshold (≈ 90 %). That is roughly **72 %** overall. This is the
price of the much smaller sets: nationally about 8.4 habitats per cell instead
of 15.7 (estimated from a regular sample of 4 million cells of
`habitat_set_size.tif`).
The flat sets of §7 (90 %) and the two-level sets of §8 are **different
products**, and figures must say which one they show.

---

## 9. Limits of interpretation

1. **Compatibility, not establishment.** All outputs describe geophysical
   compatibility. Land-use history, nutrient legacy (P and N in former farmland),
   management, dispersal and hydrological restoration are not modelled.
2. **The reference is existing nature.** Envelopes, thresholds and radii are
   learnt from today's habitat polygons. On farmland the results are an
   extrapolation, by design.
3. **Relative vs absolute.** $p(k\mid x)$ and the sets are relative comparisons
   between classes. $S_k$ and the envelope condition are the absolute measures.
4. **Small classes.** Classes with few polygons have uncertain envelopes and use
   pooled thresholds and radii. Their coverage is less reliable (§7.6).
5. **Coverage is marginal and per class.** It is not a per-cell probability
   (§7.4).

---

## 10. Outputs

| Folder | Main files |
|---|---|
| `01_merged_raster/` | `merged_class_id.tif`, `overlap_report.csv`, `feature_split.csv` |
| `02_training/` | `training_pixels.parquet`, `class_counts.csv` |
| `04_stats/` | `summary.md`, `classifier_report.txt` (in-sample RandomForest) |
| `05_model/` | `envelopes.npz`, `envelopes_train.npz`, `scaler*.json`, `model_report*.csv` |
| `06_fuzzy/` | `NN_<code>_suitability.tif`, `best_class.tif`, `best_suitability.tif`, `best_distance.tif` |
| `07_potential/` | `habitat_richness.tif`, `habitat_richness_open.tif`, `habitat_richness_sum.tif` |
| `08_holdout/` | `holdout_report.md`, `holdout_per_class.csv`, `holdout_confusion_precision.csv`, `set_*`, `group_*` |
| `09_residual/` | `residual_category.tif` (+ `.qml`), `residual_summary.csv`, `cluster_profile.csv`, radar charts |
| `10_sets/` | `set_mask.tif` (bitmask, see `set_bits.csv`), `set_size.tif`, `map_class.tif`, `best_posterior.tif`, `lr_set_size.tif` |
| `11_groups/` | `group_set_mask.tif` (+ `.qml`), `group_combinations.csv`, `best_group.tif`, `habitat_set_mask.tif` |

Earlier versions are kept for comparison: `10_sets_relative_only/`,
`11_groups_relative_only/` (no envelope condition) and `11_groups_gaussian99/`
(Gaussian envelope boundary).

## 11. Reproducibility

- **Pipeline:**
  ```
  python -m signatur <step> --config config_climate.yaml
  ```
  Steps, in order: `climateprep`, `lookup`, `rasterize`, `extract`, `stats`,
  `holdout`, `model`, `project`, `richness`, `residual`, `residualprofile`,
  `setcalib`, `sets`, `groupcalib`, `groups`.
- **Settings:** all are in the config file, including seeds
  (`analysis.random_seed = 42`), coverages (`sets.coverage = 0.9`,
  `groups.coverage = 0.8`, `groups.habitat_coverage = 0.9`) and the envelope
  calibration level (`sets.envelope_coverage = 0.98`).
- **Software:** Python 3.12 (conda-forge), with numpy, scipy, scikit-learn,
  rasterio/GDAL and pandas.

## References

- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*
  88(2), 365–411.
- Mahalanobis, P. C. (1936). On the generalised distance in statistics.
  *Proceedings of the National Institute of Sciences of India* 2, 49–55.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8),
  913–929. https://doi.org/10.1111/ecog.02881
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random
  World*. Springer. (Conformal prediction, including the Mondrian form.)
- Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction: a gentle
  introduction. *Foundations and Trends in Machine Learning* 16(4), 494–591.
- Wilcox, R. R. (2017). *Introduction to Robust Estimation and Hypothesis
  Testing*, 4th ed. Academic Press. (Winsorising.)
- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
