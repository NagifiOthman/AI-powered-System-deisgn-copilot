# AI-powered\_System-design\_copilot

Initial backend slice for the AI-powered System Design Copilot.

This version includes only:
- Python + FastAPI backend
- OpenAPI-documented endpoints
- OpenAI API orchestration for:
	- Recommended tech stack
	- Security and DevOps checklist
	- MVP vs post-MVP feature roadmap with dependency graph and estimates

RAG/PostgreSQL grounding is intentionally deferred for a later milestone.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment template and add your key:

```bash
copy .env .env
```

4. Start API server:

```bash
uvicorn app.main:app --reload
```

5. Open API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API endpoints

- `POST /api/v1/discovery-questions`
- `POST /api/v1/design-advice`
- `POST /api/v1/roadmap`
- `POST /api/v1/full-plan`
- `GET /health`

## Example request

```json
{
	"project_idea": "A SaaS platform where engineering teams generate system design docs from plain-language product requirements.",
	"qa_context": [
		{
			"question": "Expected user scale in year 1?",
			"answer": "~5k monthly active users"
		},
		{
			"question": "Any compliance requirements?",
			"answer": "SOC2 readiness"
		}
	]
}
```



