# Enterprise Workforce Intelligence

AI-powered Workforce Intelligence Platform MVP for employee attrition analytics and data-driven HR decision support.

## Current MVP
- Loads the IBM HR Analytics Employee Attrition dataset from `data/raw/employee_attrition.csv`.
- Displays workforce KPIs and attrition rate.
- Shows attrition distribution and department-level attrition analysis.
- Provides an employee data explorer with department filtering.
- Keeps raw datasets out of GitHub; place the CSV locally for the full demo.

## Project direction
The planned platform will extend this MVP with validated employee engagement/performance analytics, role requirements and skill-gap analysis, personalized upskilling recommendations, explainable attrition risk, and monitoring.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Place the dataset at:
`data/raw/employee_attrition.csv`

## Data validation findings
The current IBM HR dataset contains 1,470 employees and 35 columns. The validation performed during development found 0 missing values, 0 duplicate rows, 1,470 unique `EmployeeNumber` values, 237 attrition cases, and 1,233 non-attrition cases. Three constant columns were identified for exclusion from ML features: `EmployeeCount`, `Over18`, and `StandardHours`.

## Deployment
The repository includes a Streamlit entry point and dependency file suitable for deployment on a Streamlit-compatible hosting service such as Render. The raw Kaggle CSV is intentionally not committed; deployment should use a permitted data-loading strategy or a demo dataset supplied through the deployment environment.
