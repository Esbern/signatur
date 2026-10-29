# Review of report sections 2.7–2.10 (habitat-development potential)

Comments on the methods text in `sinatur.pdf`, checked against the code and the
production results (`output_climate/`, climate-inclusive model) on
29 October 2026. The full methodology, with all numbers, is in
[method.md](method.md). Section references below (e.g. method.md §7.3) point to
it.

## Overall assessment

The text is accurate. Every formula matches the implementation:
- the Mahalanobis distance $D_k^2$;
- $S_k = 1 - F_{\chi^2_{11}}(D_k^2)$;
- the likelihood with $|\Sigma_k|^{-1/2}$;
- the normalisation with equal priors;
- the set $\mathcal{H}(x)$ with $q_k$ and $r_k$;
- the group sum $P(g\mid x)$;
- the conditional $p(k\mid g,x)$.

There are no misunderstandings of the method. The points below are omissions
and imprecise phrasings, ordered by importance. Points 1–4 matter most, because
without them a reader can misread the maps.

---

## Points that affect how results are read

### 1. State the combined coverage of the two-level output (§2.10)

**Issue.** The text gives 80 % for groups and 90 % for habitats within groups.
A reader will easily take the habitat-level output to carry a 90 % guarantee.
It doesn't. A site's true habitat is listed only if its group is plausible
*and* the habitat passes the within-group threshold, which is roughly
0.8 × 0.9 ≈ **72 %** overall.

**Suggested addition:**
> Because a habitat is listed only within plausible groups, the overall
> probability that a site's observed habitat appears in the two-level output is
> approximately the product of the two levels (≈ 0.8 × 0.9 ≈ 72 %). This is the
> cost of the substantially smaller sets (on average about 8 habitat classes per
> cell nationally, compared with about 16 for the single-level sets).

### 2. Distinguish the two habitat-level products, and the two "best class" maps

**Issue 1: two kinds of habitat sets.** There are two different habitat-level
set products, with different thresholds:
- **single-level sets** (§2.9, 90 %, `10_sets/set_mask.tif`): about 16 classes
  per cell nationally;
- **habitat classes within plausible groups** (§2.10, `11_groups/habitat_set_mask.tif`):
  about 8 per cell.

Each figure or table should say which one it shows.

**Issue 2: two "best class" maps.** §2.8 describes `map_class.tif`, the argmax
of $p(k\mid x)$. Earlier deliverables used `best_class.tif`: smallest $D_k^2$,
i.e. largest $S_k$. These can differ, because the likelihood includes
$|\Sigma_k|^{-1/2}$, while the distance-only map favours broad envelopes.

**Suggested addition to §2.8:**
> Note that this most-probable class differs from the class with the highest
> compatibility score $S_k$ (smallest Mahalanobis distance): the likelihood also
> accounts for envelope size, so at equal distance a narrow, specialised envelope
> provides stronger evidence than a broad one.

### 3. The guarantee is per class, on average, and measured on existing habitat (§2.9)

**Issue.** "Inclusion of the observed habitat class in approximately 90 % of
held-out reference sites" is correct, but it leaves out three things a reader
needs:

| What it is | What it is not |
|---|---|
| per class (one threshold per class) | an average over all classes |
| marginal: across many sites of class *k*, about 90 % include *k* | a 90 % probability for each cell |
| measured at existing habitat sites | a measured coverage on farmland |

**Suggested addition:**
> Coverage refers to sites of each habitat class on average (it is not a
> per-cell probability) and was assessed on existing habitat polygons;
> applied to agricultural or otherwise modified land, the sets indicate the
> habitat classes whose environmental envelope is compatible with the site at
> the precision achieved on existing habitat sites.

### 4. Add the results (§2.9 and §2.10)

**Issue.** The method is described without its outcome. Suggested numbers, all
cross-validated on held-out polygons:

| Output | Observed class/group in the set | Mean set size |
|---|---|---|
| Single-level habitat sets (90 %) | 0.905 | 14.1 of 44 |
| Habitat groups (80 %) | 0.801 | 2.2 of 7 |
| Habitat classes within the true group (90 %) | 0.903 | 3.8 |
| Most probable class only (for comparison) | 0.22 | 1 |

**Points worth stating as well:**
- **Shortfalls:** seven classes with few held-out polygons fall below 85 %
  coverage: 1310, 1320, 1340, 2250, 2330, 7150, 9170.
- **National group combinations:** 120 of the 127 possible occur. The 20
  largest cover 86 % of the land.
- **Why the sets are large:** the predictors separate broad groups well, but
  often not individual classes within a group. This is consistent with a holdout
  balanced accuracy of 0.26 for the envelope model. It is a property of the
  data, not of the method.

---

## Precision and completeness

### 5. Calibration envelopes vs production envelopes (§2.7 / §2.9)

**Issue.** The text does not say that two fits of the model are used:
- the thresholds $q_k$ and radii $r_k$ are calibrated with envelopes fitted on
  the 80 % training polygons only;
- the national maps use envelopes refitted on all polygons, with the calibrated
  $q_k$ and $r_k$ transferred.

**Suggested addition (end of §2.9):**
> Thresholds and envelope limits were calibrated with envelopes fitted to the
> training polygons only; the national predictions use envelopes refitted to all
> polygons, with the calibrated thresholds and limits applied unchanged.

### 6. "Independent habitat observations" (§2.9)

**Issue.** The held-out polygons come from the same habitat mapping programme.
They are independent of model *fitting*, but they are not an independent
dataset.

**Suggested wording:**
> …calibrated against habitat polygons withheld from model fitting (a random
> 20 % of polygons per class)…

### 7. The always-included class (§2.9)

**Issue.** The text says "the highest-probability class was also retained where
the location satisfied the environmental-envelope condition". The implementation
is slightly different: the **highest-probability class among the classes whose
envelope condition is satisfied** is always included. So the set is non-empty
whenever at least one envelope contains the location, even if the overall
highest-probability class falls outside its own envelope. The same rule applies
to groups (the best admissible group) and within groups (the best admissible
class of each plausible group).

**Suggested wording:**
> The highest-probability class among those satisfying the envelope condition
> was always retained, so that the set is empty only where a location lies
> outside all calibrated envelopes.

This also affects §2.8. The most compatible class (`map_class.tif`) is chosen
among the admissible classes, falling back to the overall argmax only where no
envelope contains the cell.

### 8. What the "environmental range" is (§2.9)

**Issue.** "Outside the class's empirically calibrated environmental range" can
be read as a range per variable. It is a single **Mahalanobis radius** $r_k$:
an ellipsoid in the 11-dimensional space, as already written in the formula.

**Why it is calibrated** (worth one sentence): the theoretical χ² limit proved
far too narrow for new sites. 9.5 % of held-out cells lay outside their own
class's 99 % Gaussian envelope, against an expected 1 %.

**Suggested wording:**
> Envelope limits $r_k$ were defined as the Mahalanobis distance enclosing 98 %
> of each class's held-out cells (polygon-weighted; pooled for classes with fewer
> than ten held-out polygons), because the theoretical chi-square limit proved too
> narrow for new sites.

### 9. Cross-validation detail (§2.9)

**Issue.** Both the thresholds **and** the envelope limits were refitted in each
half, and the halves were stratified by class.

**Suggested wording:**
> …the held-out polygons were divided into two class-stratified halves; envelope
> limits and thresholds were fitted on one half and evaluated on the other, in
> both directions.

### 10. The envelope condition also applies within groups (§2.10)

**Issue.** The text mentions the condition only for groups. It also applies to
habitat classes within a plausible group: a class is listed only if the location
lies inside its own calibrated envelope.

### 11. The number of classes (§2.7)

**Issue.** The data contain 45 habitat classes. Class 6430 has only 16 cells,
below the minimum of 150 training cells, and has no envelope.

**Suggested addition:**
> Of the 45 habitat classes in the reference data, 44 had sufficient training
> cells (≥ 150) for an envelope; 6430 (16 cells) was excluded.

### 12. Group names (§2.10)

**Issue.**
- "Grasslands and meadows" leaves out the inland rock habitat (8220) included
  in that group. "Grasslands, meadows and rock outcrops" is more accurate.
- Wooded dunes (2180) are placed under forests, not coastal dunes. That is
  deliberate, but worth a footnote.

### 13. Minor wording (§2.7)

**Issue.** Ledoit–Wolf shrinkage does not reduce sensitivity to correlated
predictors as such. It gives a well-conditioned, invertible covariance estimate
when predictors are correlated and samples are moderate.

**Suggested wording:**
> …to obtain well-conditioned covariance estimates for correlated predictors and
> moderate class sample sizes (Ledoit and Wolf, 2004).

---

## Not covered by sections 2.7–2.10 (check that it is elsewhere in the report)

- **Standardisation of the predictors:** winsorising at mean ± 5 SD, then
  z-scoring.
- **The polygon-level 80/20 holdout:** its rationale, and the holdout accuracy
  of the envelope model (balanced accuracy 0.260, 11.7 × chance) and of the
  RandomForest cross-check (0.382, 17.2 ×).
- **The in-sample RandomForest value:** if quoted, it is **0.71** for the
  climate model, from cell-level cross-validation with leakage. The 0.51 in the
  older notes is not supported by the current outputs, and was probably from the
  model without climate predictors.
- **Habitat richness:** the threshold $S_k \ge 0.5$.
- **The residual analysis** of land outside all envelopes. The cluster
  interpretations (e.g. drained peatland) are hypotheses and should be presented
  as such.
- **Limits of interpretation:** compatibility, not establishment. Nutrient
  legacy, land-use history and hydrology are not modelled. See method.md §9.
