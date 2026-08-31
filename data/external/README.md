# External workforce intelligence data

This directory contains the compact O*NET capability benchmark used by the application.

The benchmark is derived from the supplied O*NET datasets:

- `occupation_data.csv` — occupation code, title and description
- `essential_skills.csv` — occupation-level skill importance/level measures
- `software_skills.csv` — occupation-to-software/technology relationships, including Hot Technology and In Demand flags

## Recommended local raw-data layout

Place the supplied source CSVs in `data/raw/` when running the full data-engineering notebooks:

```text
data/raw/
├── employee_attrition.csv
├── occupation_data.csv
├── essential_skills.csv
└── software_skills.csv
```

The application must not infer an employee's actual certification or skill ownership from O*NET. O*NET is used as an external **role requirement benchmark**. Employee-level skills should only be populated from validated employee skill/profile data.

## Why this matters

The product is not only an attrition predictor. It connects:

`workforce risk -> role requirements -> capability gaps -> development action`

This makes recommendations auditable because the role benchmark has an explicit external source rather than an arbitrary hard-coded skill list.
