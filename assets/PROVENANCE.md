# Figure assets — provenance

Each file below is a **flattened raster reproduction** of a figure prepared for the
accompanying manuscript. Rendering to a single raster is deliberate: the
vector sources embed the individual clinical panels as separate objects, and a
flattened PNG composites them so no source frame can be recovered as a separate file.

Ancillary PNG metadata was removed; each file contains only `IHDR`, `IDAT` and `IEND`.

| asset | dimensions | sha256 |
|---|---|---|
| `fig1_workflow.png` | 1800 × 1047 | `babfaa2375c376aa145acc500843a1eba712240d7e9d21d34138b90e2d15ae92` |
| `sup_fig3_label_pairs.png` | 1600 × 598 | `c8ba69e7e2c39f485f498ba39a59b031dc81c07811b61faa831215ab86d5fc65` |

## Sources

| asset | source figure file | source sha256 (prefix) |
|---|---|---|
| `fig1_workflow.png` | `figures/fig1_workflow.pdf` | `af78a80f086dc3fd…` |
| `sup_fig3_label_pairs.png` | `figures/sup_fig3_label_pairs.pdf` | `e26aca226bc04d05…` |

The manuscript masters were not modified. The vector sources, which contain the
embedded clinical rasters, are **not** included in this repository.

Re-rendered 2026-09-04 from the current manuscript masters. The previous assets came
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
