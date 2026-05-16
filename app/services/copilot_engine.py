import json
from typing import TypeVar

from fastapi import HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.knowledge.curated_knowledge import CURATED_KNOWLEDGE_BASE
from app.models import (
    DesignAdviceResponse,
    DiscoveryProgressRequest,
    DiscoveryStepResponse,
    FullPlanResponse,
    QAPair,
    RoadmapResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

DISCOVERY_QUESTIONS: list[str] = [
    "What are you trying to build ?",
    "Who are your target users and what core problem are you solving for them?",
    "What are your expected scale and performance requirements in the first 12 months?",
    "What security/compliance constraints should the system satisfy (e.g., SOC2, GDPR, HIPAA)?",
    "What is your MVP timeline and current team capacity?",
]


class CopilotEngine:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._discovery_answers: list[QAPair] = []
        self._pending_question_index: int | None = None

    def get_discovery_step(self, payload: DiscoveryProgressRequest) -> DiscoveryStepResponse:
        if payload.restart:
            self._discovery_answers = []
            self._pending_question_index = None

        total_questions = len(DISCOVERY_QUESTIONS)
        answered_count = len(self._discovery_answers)

        if answered_count > total_questions:
            raise HTTPException(
                status_code=500,
                detail="Discovery state is invalid: too many stored answers.",
            )

        if answered_count == total_questions:
            if payload.answer is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Discovery is already complete. Use restart=true to start over.",
                )
            return self._build_discovery_complete_response(answered_count)

        if payload.answer is None:
            if self._pending_question_index is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Current discovery question is already asked. "
                        "Submit an answer before requesting another question."
                    ),
                )

            self._pending_question_index = answered_count
            return self._build_discovery_question_response(
                answered_count=answered_count,
                question_index=self._pending_question_index,
            )

        if self._pending_question_index is None:
            raise HTTPException(
                status_code=400,
                detail="No active discovery question. Request a question first.",
            )

        expected_question = DISCOVERY_QUESTIONS[self._pending_question_index]
        self._discovery_answers.append(
            QAPair(question=expected_question, answer=payload.answer.strip())
        )
        self._pending_question_index = None

        answered_count = len(self._discovery_answers)
        if answered_count > total_questions:
            raise HTTPException(
                status_code=400,
                detail=f"Discovery supports exactly {total_questions} questions.",
            )

        if answered_count == total_questions:
            return self._build_discovery_complete_response(answered_count)

        self._pending_question_index = answered_count
        return self._build_discovery_question_response(
            answered_count=answered_count,
            question_index=self._pending_question_index,
        )

    def _require_completed_discovery_context(self) -> tuple[str, list[QAPair]]:
        total = len(DISCOVERY_QUESTIONS)
        answered = len(self._discovery_answers)
        if answered < total:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Complete discovery before generating plans. "
                    f"Answered {answered}/{total} questions."
                ),
            )

        project_idea = self._discovery_answers[0].answer.strip()
        return project_idea, list(self._discovery_answers)

    @staticmethod
    def _build_discovery_question_response(answered_count: int, question_index: int) -> DiscoveryStepResponse:
        question_number = question_index + 1
        question_text = DISCOVERY_QUESTIONS[question_index]
        return DiscoveryStepResponse(
            answered_count=answered_count,
            is_complete=False,
            awaiting_answer=True,
            current_question_index=question_number,
            current_question=question_text,
            next_question_index=question_number,
            next_question=question_text,
        )

    @staticmethod
    def _build_discovery_complete_response(answered_count: int) -> DiscoveryStepResponse:
        return DiscoveryStepResponse(
            answered_count=answered_count,
            is_complete=True,
            awaiting_answer=False,
            current_question_index=None,
            current_question=None,
            next_question_index=None,
            next_question=None,
        )

    async def generate_design_advice(self) -> DesignAdviceResponse:
        project_idea, qa_context = self._require_completed_discovery_context()
        grounded_prompt = self._build_grounded_user_prompt(qa_context)
        schema_hint = {
            "recommended_tech_stack": [
                {
                    "component": "Backend",
                    "recommendation": "FastAPI",
                    "rationale": "Quick iteration, async support, strong typing",
                }
            ],
            "security_and_devops_checklist": [
                {
                    "area": "security",
                    "item": "Use OIDC + RBAC",
                    "priority": "high",
                }
            ],
        }
        user_prompt = (
            "Generate system design recommendations. Return JSON only matching this shape:\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Project idea:\n"
            f"{project_idea}\n\n"
            "Grounded discovery context:\n"
            f"{grounded_prompt}\n\n"
            "Q/A context:\n"
            f"{json.dumps([item.model_dump() for item in qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(DesignAdviceResponse, data)

    async def generate_roadmap(self) -> RoadmapResponse:
        project_idea, qa_context = self._require_completed_discovery_context()
        grounded_prompt = self._build_grounded_user_prompt(qa_context)
        schema_hint = {
            "mvp_core_features": [
                {
                    "name": "Authentication",
                    "description": "Secure login and session management",
                    "effort": "medium",
                    "impact": "high",
                    "estimate_range": "3-5 days",
                }
            ],
            "post_mvp_differentiators": [
                {
                    "name": "Advanced analytics",
                    "description": "Usage and performance insights",
                    "effort": "large",
                    "impact": "medium",
                    "estimate_range": "2-4 weeks",
                }
            ],
            "dependency_graph": [
                {
                    "feature": "User profiles",
                    "depends_on": "Authentication",
                    "reason": "Profiles require verified identity context",
                }
            ],
        }
        user_prompt = (
            "Generate an MVP-first feature roadmap and prioritization plan. "
            "Return JSON only matching this shape:\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Project idea:\n"
            f"{project_idea}\n\n"
            "Grounded discovery context:\n"
            f"{grounded_prompt}\n\n"
            "Q/A context:\n"
            f"{json.dumps([item.model_dump() for item in qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(RoadmapResponse, data)

    async def generate_full_plan(self) -> FullPlanResponse:
        design_advice = await self.generate_design_advice()
        roadmap = await self.generate_roadmap()
        return FullPlanResponse(design_advice=design_advice, roadmap=roadmap)

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an expert system design copilot. "
            "Ground recommendations in practical architecture, security, and delivery patterns. "
            "Do not output markdown or code fences. Output only valid JSON. "
            f"Grounding context: {json.dumps(CURATED_KNOWLEDGE_BASE)}"
        )

    @staticmethod
    def _build_grounded_user_prompt(answers: list[QAPair]) -> str:
        if not answers:
            return (
                "Empty,no prompt provided yet"
            )

        project_idea = answers[0].answer.strip()
        lines = [f"Project intent: {project_idea}"]
        lines.append("Discovery Q/A captured so far:")
        for item in answers:
            lines.append(f"- Q: {item.question}")
            lines.append(f"  A: {item.answer}")
        return "\n".join(lines)

    async def _request_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

        try:
            completion = await self._client.chat.completions.create(
                model=settings.openai_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI call failed: {exc}") from exc

        content = completion.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="OpenAI returned empty content.")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI returned invalid JSON: {content}") from exc

    @staticmethod
    def _validate_model(model_type: type[ModelT], data: dict) -> ModelT:
        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(status_code=502, detail=f"LLM response shape validation failed: {exc}") from exc


copilot_engine = CopilotEngine()
