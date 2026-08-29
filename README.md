# Enterprise Workforce Intelligence Platform

An agentic workforce-intelligence system that combines **ML, transformer embeddings, LLM reasoning, deterministic business rules, RAG-ready knowledge retrieval, APIs, and cloud-native deployment** to help HR answer four questions:

1. Who is at elevated attrition risk?
2. What capability/skill gaps exist?
3. Which employees are realistic reskilling candidates?
4. Where should the organization retain, reskill, or hire?

## Why this architecture

The platform deliberately separates **prediction from reasoning and action**. The attrition model produces a probability; the skill engine calculates gaps; specialist agents combine those outputs; the LLM explains the evidence without owning the underlying HR decision.

## Architecture

```text
HR / Manager
    |
Streamlit UI / FastAPI
    |
LangGraph Workforce Orchestrator
    |
+----------------+----------------+----------------+
| Attrition      | Skill Gap      | Recommendation |
| Agent          | Agent          | Agent          |
+----------------+----------------+----------------+
    |
Employee Intelligence Layer
    |
+------------------+------------------+----------------+
| ML Prediction    | Transformer      | Knowledge/RAG  |
| Logistic/XGB*    | Skill Matching   | Qdrant + LLM*  |
+------------------+------------------+----------------+
    |
Guardrails / Human Review
    |
Decision: Retain | Reskill | Develop/Hire
    |
Kubernetes + HPA + CI/CD + Monitoring
```

`*` XGBoost and RAG are extension points; they are not claimed as implemented until the corresponding data/model pipeline is validated.

## Implemented now

- IBM HR Analytics attrition dataset validation
- Stratified train/test split
- Class-balanced Logistic Regression baseline
- Precision, recall and ROC-AUC evaluation
- Employee-level probability/risk scoring
- Transparent skill-gap engine
- Readiness scoring
- Specialist Attrition, Skill Gap and Recommendation agents
- LangGraph orchestration wrapper
- Transformer-based semantic skill matching with `all-MiniLM-L6-v2`
- Optional LLM explanation agent using the OpenAI Responses API
- FastAPI endpoints with Pydantic request validation
- Streamlit dashboard
- Dockerfile
- Kubernetes namespace, API deployment/service and HPA
- GitHub Actions tests + container build
- Unit tests for skill gaps, readiness, prediction probability and orchestration

## Data validation

The IBM HR dataset contains **1,470 employees and 35 columns**. Validation found 0 missing values, 0 duplicate rows, 1,470 unique `EmployeeNumber` values, 237 attrition cases (16.12%), and 1,233 non-attrition cases (83.88%). `EmployeeNumber` and constant columns are excluded from model features.

## Agent responsibilities

### Attrition Agent
Calls the validated ML pipeline and returns probability + risk level.

### Skill Gap Agent
Maps a role to required capabilities, compares them with supplied employee skills, and calculates readiness.

### Recommendation Agent
Applies transparent business rules to produce a retain/reskill/develop-or-hire recommendation.

### Workforce Orchestrator
Coordinates specialist agents. The LangGraph wrapper provides an explicit stateful orchestration layer.

### LLM Explanation Agent
Uses an LLM only for evidence-grounded explanation and human-review questions. It does **not** replace the deterministic prediction or business rules.

### Transformer Skill Matcher
Uses sentence-transformer embeddings and cosine similarity to recognize semantically related skills rather than relying only on exact string matches.

## Deployment

### Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

FastAPI:

```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t workforce-ai .
docker run -p 8501:8501 workforce-ai
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/hpa.yaml
```

Create the LLM secret without committing it:

```bash
kubectl -n workforce-ai create secret generic workforce-secrets --from-literal=openai-api-key="$OPENAI_API_KEY"
```

## Important data limitation

A genuine employee skill-gap system requires a current employee-skill inventory. If the supplied datasets do not contain employee-level current skills, the project must use a clearly labelled controlled MVP skill table rather than inventing real employee skills. Role requirements can be sourced from O*NET and the supplied role/skill datasets after their schemas and join keys are validated.

## Roadmap

1. Validate all five raw datasets and their relationships
2. Add engagement/performance intelligence
3. Add O*NET role taxonomy and required skills
4. Establish employee-skill inventory
5. Compare Logistic Regression, Random Forest and XGBoost
6. Add SHAP explainability
7. Add RAG over HR policy/training/role knowledge
8. Connect Qdrant + Redis + PostgreSQL
9. Expand LangGraph multi-agent workflows
10. Add model/data drift monitoring
11. Add Prometheus/Grafana/OpenTelemetry
12. Deploy the services to Kubernetes with HPA and CI/CD

## Responsible-use boundary

The system is decision support, not an autonomous employment decision-maker. Risk scores and recommendations require human review. Sensitive attributes should not be used as decision features, and the LLM is constrained to explaining structured evidence rather than inventing HR facts.
