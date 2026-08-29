# Enterprise Workforce Intelligence

AI-powered Workforce Intelligence Platform MVP for data-driven HR decision support.

## What is implemented
- IBM HR Analytics Employee Attrition dataset validation
- Attrition analytics and department-level analysis
- Stratified train/test split
- Class-balanced Logistic Regression pipeline
- One-hot encoding for categorical features
- Median imputation and standardization inside the ML pipeline
- Employee-level attrition-risk scoring
- Interactive Streamlit dashboard
- Render deployment configuration

## Data validation findings
The IBM HR dataset used during development contains **1,470 employees and 35 columns**. Validation found:

- 0 missing values
- 0 duplicate rows
- 1,470 unique `EmployeeNumber` values
- 237 attrition cases (16.12%)
- 1,233 non-attrition cases (83.88%)
- `Age` range: 18–60
- Satisfaction/involvement scales observed: 1–4
- Constant columns identified: `EmployeeCount`, `Over18`, `StandardHours`

The three constant columns and `EmployeeNumber` are excluded from model features. `EmployeeNumber` remains available for employee identification.

## Current model
The first model is deliberately a **class-balanced Logistic Regression** baseline because it is fast, interpretable, probability-producing, and appropriate for an MVP. The target is imbalanced, so the dashboard reports precision, recall, and ROC-AUC rather than relying on accuracy alone.

The model is trained at app startup from the supplied dataset. No fabricated model metrics are stored in the repository.

## Local setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Place the dataset here:

```text
data/raw/employee_attrition.csv
```

## Repository structure

```text
enterprise_workforce_intelligence/
├── data/
│   └── raw/
│       └── employee_attrition.csv   # supplied locally / deployment environment
├── src/
│   ├── __init__.py
│   └── train_model.py
├── streamlit_app.py
├── requirements.txt
├── render.yaml
└── README.md
```

## Deployment

`render.yaml` is configured for a Python web service running Streamlit on Render. The deployment requires the dataset to be available at `data/raw/employee_attrition.csv` because the current MVP trains from that data at startup.

For a public repository, do not commit private/proprietary HR data. Use only a dataset you are permitted to redistribute.

## Planned next modules

1. Employee engagement/performance integration after validating join keys
2. O*NET role requirements
3. Employee skill inventory
4. Deterministic skill-gap engine
5. Upskilling recommendation layer
6. Explainability and monitoring
7. Broader workforce intelligence dashboard
