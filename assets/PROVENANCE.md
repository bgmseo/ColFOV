# Figure assets — provenance

Each file below is a **flattened raster reproduction** of a figure prepared for the
accompanying manuscript. Rendering to a single raster is deliberate: the
vector sources embed the individual clinical panels as separate objects, and a
flattened PNG composites them so no source frame can be recovered as a separate file.

Ancillary PNG metadata was removed; each file contains only `IHDR`, `IDAT` and `IEND`.

| asset | dimensions | sha256 |
|---|---|---|
| `fig1_workflow.png` | 1800 × 1047 | `babfaa2375c376aa145acc500843a1eba712240d7e9d21d34138b90e2d15ae92` |
| `sup_fig3_label_pairs.png` | 1600 × 598 | `40b24d6c84cc74f7e20be42cfc51c87f1f614899b11be5286cf8883dad4b2392` |

## Sources

| asset | source figure file | source sha256 (prefix) |
|---|---|---|
| `fig1_workflow.png` | `figures/fig1_workflow.pdf` | `af78a80f086dc3fd…` |
| `sup_fig3_label_pairs.png` | `figures/candidates/git_sup_fig3_label_pairs.pdf` | `6840ea89bdb1d465…` |

The manuscript masters were not modified. The vector sources, which contain the
embedded clinical rasters, are **not** included in this repository.

The label-pair asset is the repository-specific variant (`git_` prefix), which carries the
"Three b-boxes from semantic segmentation" panel and the Full-FOV / Inner-FOV / PiP legend;
the manuscript's own supplementary figure is a different layout and is not used here.

Re-rendered 2026-09-04 from the current masters. The previous assets came
from a superseded working export whose embedded rasters were placed off their native
aspect ratio (25 of 25, worst 12.6 % too tall), which misstated the very geometry the
figure exists to show. That was corrected in the Illustrator master on 2026-09-03.

## Conditions recorded at the time of inclusion

- Inclusion, including the clinical sample panels composited within them, was approved
  by the author for these exact manuscript figures only.
- No other clinical image, qualitative example, or newly rendered panel is included.
- Whether this reproduction is compatible with the hospital data-use agreement, the
  IRB approval and any journal copyright transfer remains the responsibility of the
  research team; publication of the figure in a journal does not by itself establish a
  right to redistribute it here.
