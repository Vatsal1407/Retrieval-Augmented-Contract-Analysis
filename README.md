# Legal Contract Analyzer

Agentic RAG system for contract intelligence with hybrid retrieval, tool-routing, evaluation, monitoring, and production deployment.

Tech stack: Python 3.11, LangChain, Chroma + optional BM25 rerank, FastAPI, Streamlit, RAGAs-style evaluation, SQLite/PostgreSQL, Docker, AWS ECS Fargate, Terraform, GitHub Actions.

## Live Demo

- App URL: set your ALB DNS name after first ECS deploy
- Monitoring URL: `${ALB_DNS}/dashboard/`

## Resume Bullet

Built a Legal Contract Analyzer using Agentic RAG (LangChain, Chroma + BM25 rerank, HuggingFace/Ollama) on CUAD (510 contracts); implemented scoped multi-contract retrieval, evaluated with faithfulness/relevance/precision/recall signals, and deployed on AWS ECS Fargate with real-time monitoring.

## Problem Statement

Contract review is slow and expensive. This project automates clause discovery and grounded Q&A while explicitly addressing common RAG failure modes:

1. Wrong retrieval when answers span multiple clauses.
2. No fallback when the document lacks the answer.
3. No measurable quality signal.

This system solves these with hybrid retrieval (BM25 + dense), agentic tool routing (contract vs web), and RAGAs metric logging.

## Dataset (CUAD)

- Source: HuggingFace (`theatticusproject/cuad` with `cuad` fallback)
- Scale: 510 real contracts, 13k+ annotations, 41 clause categories
- Why CUAD matters: enables retrieval and answer quality evaluation against grounded legal spans

Load example:

```python
from datasets import load_dataset

ds = load_dataset("theatticusproject/cuad")

# If split-size verification fails in your environment:
# ds = load_dataset("theatticusproject/cuad", verification_mode="no_checks")
```

Note: this dataset variant exposes a PDF feature column in some environments. The loader in this project extracts contract text from those PDF rows automatically during ingestion.

## Architecture

```mermaid
flowchart TD
		A[Streamlit Frontend] --> B[FastAPI /ask]
		B --> C[ContractQAPipeline]
		C --> D[Clause-Aware Retriever]
		D --> D1[Chroma Dense Search]
		D --> D2[Optional BM25 Rerank]
		D2 --> E[Answer Generator]
		E --> B
		B --> F[Evaluator]
		F --> G[(SQLite or PostgreSQL)]
		G --> H[Streamlit Monitoring Dashboard]
```

## Why Hybrid Retrieval

- Dense retrieval handles semantic paraphrases.
- BM25 captures exact legal keywords and section references.
- Reciprocal Rank Fusion (RRF) merges heterogeneous ranking outputs without fragile score normalization.

RRF formula:

$$
	ext{RRF}(d) = \sum_i \frac{1}{rank_i(d) + 60}
$$

## Project Structure

```text
legal-contract-analyzer/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Makefile
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval_samples/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── artifact_store.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── answerer.py
│   │   ├── contracts_registry.py
│   │   └── chat_scope_registry.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── prompts.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_evaluator.py
│   │   ├── retrieval_metrics.py
│   │   ├── run_eval.py
│   │   └── metrics_store.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── ask.py
│   │   │   ├── query.py
│   │   │   ├── contracts.py
│   │   │   ├── upload.py
│   │   │   ├── delete.py
│   │   │   └── metrics.py
│   │   └── schemas.py
│   └── monitoring/
│       ├── __init__.py
│       └── dashboard.py
├── frontend/
│   └── app.py
├── tests/
│   ├── test_retrieval.py
│   ├── test_agent.py
│   ├── test_api.py
│   ├── test_eval.py
│   ├── test_security.py
│   ├── test_chunker.py
│   └── test_ingestion_embedder.py
├── infra/
│   ├── Dockerfile
│   └── terraform/
│       ├── main.tf
│       ├── ecr.tf
│       ├── rds.tf
│       ├── alb.tf
│       └── variables.tf
└── .github/
		└── workflows/
				└── ci-cd.yml
```

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Configure environment.

```bash
cp .env.example .env
```

3. Build retrieval artifacts from CUAD.

```bash
make ingest
```

4. Start local stack.

```bash
make run
```

5. Open services.

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`
- Dashboard: `http://localhost:8502`

## API Endpoints

- `POST /ask`
	- Input: `{ "query": "...", "chat_id": "...", "contract_id": "optional" }`
	- Output: answer, source chunks, citations, sources, tool used, routing reason, evaluation
- `POST /query`
	- Legacy-compatible query endpoint for answer + citations payload
- `POST /upload`
	- Upload `.txt` or `.pdf`; returns `chat_id` and indexed contract ids
	- Rejects unsupported file types (HTTP 415) and oversized files (HTTP 413)
- `GET /contracts`
	- Lists contracts (filter by `chat_id` for scoped visibility)
- `DELETE /contracts/{contract_id}?chat_id=...`
	- Deletes a contract from storage and vector database (requires ownership via `chat_id`)
- `GET /metrics`
	- Returns recent metric rows, trends, and routing analytics
- Optional auth:
	- Set `API_AUTH_TOKEN` and provide `x-api-key` header from clients

## Evaluation (RAGAs)

Batch flow:

1. Build sample set (`data/eval_samples/`).
2. Run evaluator.
3. Persist metrics to PostgreSQL.
4. Visualize trends in dashboard.

Run:

```bash
# Standard evaluation
make eval

# Dense vs hybrid comparison with retrieval metrics (Precision@K, Recall@K, MRR, nDCG@K)
make eval-compare

# Full evaluation including unanswerable questions for hallucination resistance
make eval-full
```

> **Note:** No pre-computed benchmark numbers are included in this README. Run `make eval-full` on
> your own environment to generate reproducible results saved to `eval_results/`.

## Monitoring Dashboard

Dashboard features:

- Metric trend lines over configurable time windows
- Last N query table with tool routing and fallback flags
- Faithfulness threshold alert (red below 0.90)
- Query analytics by tool usage frequency

## Docker and Local Orchestration

- Multi-stage Docker build in `infra/Dockerfile`
- `docker-compose.yml` runs:
	- FastAPI backend
	- Streamlit user app
	- Streamlit monitoring app
	- PostgreSQL

By default, metrics use SQLite unless `DATABASE_URL` points to PostgreSQL.

## AWS Deployment (ECS Fargate)

Provisioned by Terraform:

- ECR repos for API/dashboard images
- ECS cluster and Fargate services
- ALB path routing (dashboard available on `/dashboard/*`)
- Optional HTTPS listener when `acm_certificate_arn` is configured
- RDS PostgreSQL for metric logs
- S3 bucket for artifact storage
- Secrets Manager for API auth token, Tavily, and Hugging Face token

Important production decision:

- Runtime retrieval uses persisted Chroma collections; ingestion refreshes per-contract chunks and metadata.
- Optional strict tenant-style scoping can be enabled with `REQUIRE_CHAT_SCOPE=1`.
- Contract and chat-scope registries support DB-backed persistence via `REGISTRY_BACKEND=auto|db|file` (use `db` with shared PostgreSQL in multi-instance deployments).
- Uploaded raw contract text and retrieval chunk metadata support DB-backed persistence via `ARTIFACT_STORE_BACKEND=auto|db|file`.
- When shared artifact storage is enabled, new instances can bootstrap local vector state from DB-backed chunks, keeping query behavior instance-independent.
- Existing warm instances can periodically refresh local vector state from shared artifacts using `VECTOR_ARTIFACT_SYNC_INTERVAL_SECONDS`.
- Run schema migrations with `alembic upgrade head` before starting API services in production.
- Set `DB_AUTO_CREATE_TABLES=0` in production after migrations are managed through Alembic.

## CI/CD

Workflow: `.github/workflows/ci-cd.yml`

On push to `main`:

1. Run tests.
2. Build and push API/dashboard images to ECR.
3. Trigger ECS rolling redeploy.

Required GitHub secrets:

- `AWS_REGION`
- `AWS_ROLE_TO_ASSUME`
- `ECR_REPOSITORY_API`
- `ECR_REPOSITORY_DASHBOARD`
- `ECS_CLUSTER`
- `ECS_SERVICE_API`
- `ECS_SERVICE_DASHBOARD`

## Security

- **Mandatory chat scope:** `REQUIRE_CHAT_SCOPE=1` (default) enforces per-chat contract isolation. One user's contracts are never visible to another chat session.
- **File type validation:** Upload endpoint rejects unsupported file types (only `.pdf`, `.txt`, `.text` accepted) with HTTP 415.
- **File size limits:** Uploads over 10 MB are rejected with HTTP 413.
- **Prompt injection sanitization:** Retrieved context chunks are stripped of common injection patterns (`[INST]`, `<<SYS>>`, role overrides, instruction forgetting) before prompt insertion.
- **Document deletion:** `DELETE /contracts/{contract_id}?chat_id=...` removes data from vector store, artifact store, and registry.
- **API authentication:** Optional `API_AUTH_TOKEN` environment variable enables API key authentication via `x-api-key` header.
- **Sanitized error responses:** Internal error details are never leaked to API clients.

> **Disclaimer:** Generated answers are not legal advice. This system does not replace professional legal review.

## Design Decisions

1. RAG over fine-tuning:
	 legal documents change frequently, so externalized retrieval is cheaper and easier to update.

2. Hybrid over dense-only:
	 legal wording has both semantic variance and exact term sensitivity.

3. Agentic routing:
	 LLM-based tool routing with contract search and web fallback, activated via `USE_AGENT_ROUTING=1`. When disabled (default), the pipeline path handles queries directly.

4. Measurable quality:
	 RAGAs + logged trend lines catch regressions before production incidents.

## Sample Queries

- What is the indemnification limit in this contract?
- Is there a termination for convenience clause?
- What obligations survive termination?
- Compare liability cap language between two uploaded contracts.
- What is the typical indemnity cap in SaaS deals? (web fallback)

## Development Commands

```bash
make install
make ingest
make api
make frontend
make dashboard
make eval            # Standard evaluation
make eval-compare    # Dense vs hybrid retrieval comparison
make eval-full       # Full evaluation with unanswerable questions
make test
make migrate
```

Migration notes:

- `make migrate` applies Alembic migrations to the database from `DATABASE_URL`.
- Keep `DB_AUTO_CREATE_TABLES=1` for local SQLite convenience.
- Set `DB_AUTO_CREATE_TABLES=0` in shared environments so schema lifecycle is migration-driven.
