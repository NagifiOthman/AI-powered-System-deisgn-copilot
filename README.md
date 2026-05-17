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
- `POST /api/v1/requirements-context`
- `POST /api/v1/design-advice`
- `POST /api/v1/roadmap`
- `POST /api/v1/full-plan`
- `GET /health`

## Discovery flow example (sequential)

1) Ask for the first discovery question:

```json
{}
```

2) Submit one answer for the currently asked question:

```json
{
	"answer": "An AI copilot that generates system design plans from product ideas."
}
```

Repeat step 2 until `is_complete` is `true`.

Important Notes:
- The backend controls question order and index progression.
- If you request another question before answering the current one, you get a validation error.
- Discovery answers are validated by the LLM; out-of-context answers are rejected with 400.
- To restart discovery from question 1, call:

```json
{
	"restart": true
}
```

## Requirements context (after discovery)

After discovery is complete, submit functional and non-functional requirements:

```json
{
	"functional_requirements": [
		"Users can create and save system design projects",
		"Users can generate architecture recommendations from product briefs"
	],
	"non_functional_requirements": [
		"P95 API latency under 2 seconds for design generation requests",
		"SOC2-ready audit logging for user and admin actions"
	]
}
```

This payload is validated by the LLM and normalized before being stored.

## Planning generation after discovery

After all 5 discovery answers are submitted **and** requirements context is accepted,
generate outputs without re-sending `qa_context`:

- `POST /api/v1/design-advice`
- `POST /api/v1/roadmap`
- `POST /api/v1/full-plan`

The backend uses the stored discovery Q/A context as the single source of truth.

## Example discovery-derived context

```json
{
	"qa_context": [
		{
			"question": "What are you trying to build ?",
			"answer": "An AI copilot that generates system design plans from product ideas."
		},
		{
			"question": "Who are your target users and what core problem are you solving for them?",
			"answer": "Startup teams who need fast and reliable architecture guidance."
		},
		{
			"question": "What are your expected scale and performance requirements in the first 12 months?",
			"answer": "About 10k monthly users and low-latency API responses under 2 seconds."
		},
		{
			"question": "What security/compliance constraints should the system satisfy (e.g., SOC2, GDPR, HIPAA)?",
			"answer": "SOC2 readiness and GDPR baseline controls."
		},
		{
			"question": "What is your MVP timeline and current team capacity?",
			"answer": "An 8-week MVP with 2 backend and 1 frontend engineer."
		}
	]
}
```



