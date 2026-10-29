# Plausible-class sets: replacing "the best habitat" with "the habitats we cannot rule out"

Steps `setcalib` and `sets` · module `src/signatur/sets.py` · outputs in
`08_holdout/` and `10_sets/` · climate model (`config_climate.yaml`)

## 1. The problem with `best_class.tif`

`best_class.tif` gives every cell the single habitat type whose envelope it sits
closest to. Two very different situations end up looking the same on that map:

- **Clear winner:** a cell is typical of habitat 9130 and unlike every other type.
- **Near tie:** a cell sits almost equally close to 9130, 9160 and 6210. 9130
  happens to win by a hair.

Both are coloured "9130". The obvious fix, showing the best three, replaces one
arbitrary number (1) with another (3). What we want is a set that **adapts to
each cell**: one class where the data clearly points to one, many where it
doesn't. The size of that set should come from the data, not from us.

## 2. Step 1: how much more does the cell look like k than like the others?

The existing suitability value (the chi-square value from the Mahalanobis
distance) is **absolute**. It asks "is this cell a typical member of habitat k?",
and answers each habitat separately. A cell can be a typical member of ten
habitats at once, and that is exactly what the richness map counts.

To choose *between* habitats we need a **relative** measure. Each habitat's
envelope is a multivariate Gaussian, so it gives a likelihood for every cell:

    log L_k(x) = −½ · d²_k(x) + ½ · log |precision_k|

The second term matters. A narrow envelope (a specialised habitat) packs its
probability into a small region, so a cell inside it is stronger evidence than
a cell inside a broad envelope. Normalising across habitats gives a posterior
probability:

    p(k | x) = L_k(x) / Σ_j L_j(x)

This assumes **equal priors**: every habitat type is treated as equally likely
in advance, whatever its current area. That is the right choice for a potential
map; we don't want common habitats to win just because they are common.

Read p(k | x) as *"of all the habitats, how much of the evidence points to k?"*
The values over all habitats in a cell sum to 1.

## 3. Step 2: where do we cut? This is where the held-out polygons come in

Now we need a rule: *include habitat k in the set if p(k | x) is at least q.*
Which q? Too high, and the true habitat is often left out. Too low, and every
set contains everything. We want q such that **the true habitat is in the set
for, say, 90 % of sites**.

We can only check that at sites where we **know** the true habitat, and the
answer is only honest if the model **has not seen those sites**. That is what
the held-out polygons provide.

### What the held-out polygons are

In `rasterize`, the habitat polygons of each class were split once, at random:
80 % for **training** and 20 % **held out** (`split == 1`). The held-out
polygons are surveyed nature areas with a known habitat type, but the envelopes
used for calibration were fitted **without them**. From the model's point of
view they are new sites, like a field it has never seen.

### The calibration itself, step by step

1. Fit the envelopes on the **training** polygons only (`envelopes_train.npz`).
2. For every cell in the **held-out** polygons, compute p(k | x) for all
   habitats, and note the probability given to the **correct** habitat, the one
   that is actually there. Call it p_true.
3. Some cells give the correct habitat a high p_true (the model "got it");
   others a very low one (the model was fooled). Sort them.
4. Find the value q that **90 % of p_true values are above**. That is the
   cutoff.
5. By construction, a set built as {k : p(k | x) ≥ q} contains the true habitat
   in 90 % of those held-out cells. New sites behave statistically like the
   held-out ones, so the same holds for them. That last step is the whole
   argument, and it is why the polygons have to be **unseen**.

An analogy: you are setting the size of a fishing net. You can't choose the
mesh by looking at the fish you trained on; you need a fresh catch where you
know what you were trying to catch. You adjust the net until it holds 90 % of
that fresh catch. A larger net (a bigger set) is the cost of being sure.

This procedure is called **split conformal prediction**. Its attraction is that
the 90 % does not depend on the Gaussian envelopes being correct. Even if an
envelope is a poor shape, the cutoff is checked against reality, and the sets
simply grow to compensate.

### Why training cells would not work

Cells inside the training polygons helped fit the envelopes, so the model is
over-confident on them: it gives their true habitat a higher probability than
it would give a new site. A cutoff calibrated on them would be too optimistic,
the sets too small, and the real coverage on new sites below 90 %. The
earlier holdout analysis (`08_holdout/holdout_report.md`) showed this effect
for top-1 accuracy.

### Three refinements

- **One polygon, one vote.** Cells in the same polygon are nearly identical
  geophysically, so a 500-ha polygon would otherwise outvote fifty small ones.
  Each polygon gets equal total weight. That makes "90 % of sites" mean 90 % of
  *sites*, not 90 % of *hectares*.
- **One cutoff per habitat (Mondrian).** With a single shared cutoff, the large,
  easy classes would achieve more than 90 % and the small, hard ones less, while
  the average still came out at 90 %. Calibrating each habitat separately gives
  about 90 % **for every habitat type**. Classes with fewer than 10 held-out
  polygons don't have enough data for their own cutoff; they use the shared
  (pooled) one, and their coverage is less reliable (see the table below).
- **The top class is always included**, so the set is never empty. Cells
  outside *every* envelope (best suitability below 0.05, as in the `residual`
  step) get set size **0**, meaning "no surveyed habitat fits". A relative
  measure would otherwise still name a best class there.

### How we check that it works

Calibrating on the held-out polygons and then measuring coverage on the same
polygons would be circular. So `setcalib` splits the held-out polygons into
two halves. It calibrates on half A and measures on half B, then the reverse.
Those are the coverage numbers reported below. The cutoffs used for the final
map are then fitted on all held-out polygons.

One subtlety: the cutoffs are calibrated with the 80 % envelopes but applied
with the production envelopes, which are fitted on 100 % of the polygons. The
production envelopes are slightly better estimated, so if anything the real
coverage is a little above the target. This is standard practice.

## 4. What came out (climate model)

From `08_holdout/set_report.md` (5,047 held-out polygons, 44 habitat types):

| rule | true habitat in the set | mean set size |
|---|---|---|
| **conformal, 90 %** | **90.5 %** | **14.1** |
| likelihood-ratio rule (within 1/8 of the best) | 46 % | 2.9 |
| best class only | 22 % | 1 |

### What this means

- **The guarantee holds.** Measured on polygons the model never saw, the true
  habitat is in the set 90.5 % of the time, against a target of 90 %.
- **But the sets are large: 14 of 44 habitats on average.** That is the price
  of the guarantee, and it is a **finding about the data**, not a flaw in the
  method. The 11 geophysical PCs separate *groups* of habitats (coastal,
  heath/acid, rich forest, wetland, …) well, but they often cannot separate
  individual types *within* a group. This matches the earlier holdout top-1
  accuracy of about 26 %. Other factors that determine the exact habitat type,
  such as land-use history, management, grazing and seed sources, are not in
  the predictors.
- **"Best three" would mislead.** Even the likelihood-ratio rule, with about 3
  classes per cell, misses the true habitat more than half the time. A top-3
  map would look precise without being accurate.
- **The set-size map is itself a result.** Where sets are small, the geophysics
  clearly identifies the habitat type. Where they are large, the location is
  geophysically "generic", and the choice of habitat there depends on things
  other than the geophysics.

### Classes to treat with care

These habitat types fall short of 85 % coverage when cross-validated. All have
few held-out polygons (2–14), so their cutoff is either pooled or estimated from
very little data: **1310, 1320, 1340, 2250, 2330, 7150, 9170**.
Per-class numbers are in `08_holdout/set_coverage.csv`.

## 5. Implications to keep in mind

1. **The guarantee is about sites on average, not about each cell.** "90 %"
   means that across many sites of habitat k, 90 % have k in their set. It does
   not mean "this particular cell has a 90 % chance".
2. **It is measured on existing nature.** The held-out polygons are surveyed
   habitat sites. Most of Denmark is farmland, forestry or built-up land, so for
   those cells the set says *"these are the habitats whose geophysical
   conditions this cell matches, at the precision we can achieve at real
   habitat sites"*. That is the right reading for a potential map, but it is
   an extrapolation, not a measured coverage for farmland.
3. **The coverage level is a choice, but an interpretable one.** 90 % is a
   convention. 80 % gives smaller sets and misses more often. Unlike "top 3",
   the trade-off can be stated precisely. Changing it means re-running
   `setcalib` and `sets` with `sets.coverage` in the config (about 70 minutes).
4. **Equal priors** make it a statement about the geophysics only. It does not
   favour habitats that are common today.
5. **`map_class.tif` differs from `best_class.tif`.** The new best class uses
   the posterior, which includes the envelope width, while the old one used
   distance only. Broad envelopes win less often under the new one. Use
   `map_class` together with `best_posterior` for "how clearly the winner
   wins".

## 6. Outputs and how to read them

`10_sets/` (all on the standard 10 m grid):

| file | type | content |
|---|---|---|
| `set_size.tif` | int16 | number of plausible habitats; 0 = outside every envelope; −1 = no data |
| `set_mask.tif` | uint64 | which habitats: bit *j* set means the habitat in row *j* of `set_bits.csv` is plausible |
| `set_bits.csv` | csv | bit → class_id, naturkode, name, main habitat type |
| `best_posterior.tif` | float32 | p(best habitat \| x): close to 1 = clear winner, low = near tie |
| `map_class.tif` | int16 | habitat with the highest posterior |
| `lr_set_size.tif` | int16 | set size under the likelihood-ratio rule, for comparison |
| `set_size_summary.csv` | csv | cells, ha and share per set size |

`08_holdout/`: `set_calibration.csv` (the cutoff for each habitat),
`set_coverage.csv` (cross-validated coverage and set size for each habitat),
`set_report.md`.

To test whether habitat *j* is plausible in QGIS's raster calculator:
`floor("set_mask@1" / 2^j) % 2 = 1`. In Python: `(mask >> j) & 1`.

The mask is also the input for the combination analysis (an UpSet plot of the
most common sets).

## 7. Two levels: habitat groups first, then habitats within them

Steps `groupcalib` and `groups` · module `src/signatur/groups.py` · outputs in
`08_holdout/group_*` and `11_groups/`

With sets of about 14 habitats per cell, there are tens of thousands of
distinct combinations. The 44 habitats are therefore grouped into the 7 classes
in `data/types and classes.csv` (`Klasse nr`): Strande og strandenge,
Kystklitter, Indlandsklitter, Heder og krat, Overdrev/eng/klipper, Moser and
Skove. The grouping comes from that file, not from the first two digits of the
code, so 2180 Skovklit sits under Skove.

The method is the same, one level up:

- **Group posterior** = the sum of the posteriors of a group's habitats. This
  is "how much of the evidence points to forest at all".
- **Group set** = every group whose posterior clears its cutoff, calibrated on
  the held-out polygons at **80 %** coverage (`groups.coverage`).
- **Habitats within a group:** for each plausible group, the habitats whose
  posterior *within the group*, p(k | x) / P(group | x), clears a cutoff. That
  cutoff is calibrated at **90 %** (`groups.habitat_coverage`) on held-out cells
  whose true group is that group.

Just picking the best group would bring back the winner-takes-all problem: the
best group is the right one for only 56 % of held-out sites.

### Gaps between niches: the member-envelope condition

The posterior is relative: it says which group a cell looks most like, not
whether the cell lies inside the group's niches. Take a cell in the gap between
two forest habitats, outside both envelopes but even further from everything
else. At habitat level the evidence is split between the two. At group level
the shares are added, and "Skove" looks certain. (The group is a mixture of its
habitats' envelopes, not one large envelope, so the group itself does not fill
the gap. The relative comparison does.) The calibration cannot catch this,
because held-out sites are existing habitat and hardly ever lie in such gaps.

In a sample of 2 million land cells, 2.5 % of the plausible groups had no
habitat with suitability ≥ 0.05. The worst affected were Kystklitter (5.4 %)
and Indlandsklitter (4.9 %).

The rule is therefore (`sets.member_envelope`, on by default, used by both
`sets` and `groups`):

- a habitat is listed only if the cell is inside its own envelope;
- a group is plausible only if at least one of its habitats meets that;
- a cell inside no envelope at all gets the empty set (size 0).

### The envelope boundary is calibrated too

"Inside its own envelope" was first defined from the Gaussian: suitability ≥
0.01, the 99 % envelope. On the held-out polygons that turned out to be far too
narrow. **9.5 % of new sites lie outside their own habitat's 99 % envelope**,
where the Gaussian predicts 1 %. Variation between sites is larger than within
the training polygons. For 16 of the 44 habitats more than 10 % of new sites
fell outside, so the 90 % target could not be reached, and the sets grew to
22.7 habitats.

So the boundary is calibrated on the held-out polygons, just like the cutoff.
For each habitat, the radius (in Mahalanobis d²) is the one that contains
**98 %** of the habitat's unseen sites (`sets.envelope_coverage`). Each polygon
counts once, and habitats with fewer than 10 held-out polygons use a shared
radius. Radius and cutoff are both cross-validated: fitted on one half of the
held-out polygons and evaluated on the other.

- **Wider than the Gaussian.** The calibrated radii are a median 2.5 times
  the Gaussian 99 % radius in d², and much larger for some habitats. The real
  niches at new sites are wider than the training polygons suggest, and some of
  the "gaps" seen with Gaussian envelopes are conditions where the habitat
  actually occurs.
- **98 %, not 99 %.** At 99 % the radius is set by one or two extreme
  polygons: for 6210 Kalkoverdrev it became 15,915 times the Gaussian, which
  in practice switches the rule off. At 98 % the large habitats have stable
  radii (6210: 4.9 times).
- **Cost.** 1.5 % of held-out sites fall outside their own calibrated envelope.
  The cutoffs make up for this, so coverage stays at the target.

Earlier versions are kept for comparison: `11_groups_relative_only/` and
`10_sets_relative_only/` (no condition), and `11_groups_gaussian99/` (Gaussian
boundary at 0.01).

Results (cross-validated on held-out polygons, `08_holdout/group_report.md`,
with the calibrated condition):

| level | true one in the set | mean set size |
|---|---|---|
| habitat, all 44 (90 %) | 90.5 % | 14.1 |
| group (80 %) | 80.1 % | 2.2 of 7 |
| habitat within its true group (90 %) | 90.3 % | 3.8 |

Across the held-out sites, 28 % have one plausible group, 35 % two, 27 % three
and 11 % four or more. With 7 groups there are at most 127 combinations, so the
group map can be read directly, and an UpSet plot or Sankey diagram is feasible.

Combined coverage: a site's true habitat appears in `habitat_set_mask` when its
group is in the group set (80 %) *and* the habitat passes its within-group
cutoff (90 %). That gives roughly 0.8 × 0.9 ≈ 72 % overall. This is the price of
the much smaller sets, about 7 habitats per cell instead of 14–16.

`11_groups/`:

| file | type | content |
|---|---|---|
| `group_set_mask.tif` | uint8 | bit g−1 set = group g plausible (`group_bits.csv`); 0 = outside every envelope |
| `group_set_size.tif` | int16 | number of plausible groups |
| `best_group.tif` / `best_group_posterior.tif` | uint8 / float32 | most likely group and how clearly it wins |
| `habitat_set_mask.tif` | uint64 | plausible habitats inside the plausible groups (bits as `10_sets/set_bits.csv`) |
| `habitat_set_size.tif` | int16 | number of those habitats |
| `group_combinations.csv` | csv | every group combination present, with cells, ha and share, largest first |

Because `group_set_mask.tif` takes at most 128 values, it can be styled
directly in QGIS as a paletted/unique-values raster. The names for each value
are in `group_combinations.csv`.
