import json
from typing import TypeVar

from fastapi import HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.knowledge.curated_knowledge import CURATED_KNOWLEDGE_BASE
from app.models import (
    CopilotRequest,
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

    def get_discovery_step(self, payload: DiscoveryProgressRequest) -> DiscoveryStepResponse:
        index = payload.question_index

        if index == 0:
            return self._build_discovery_response(answered_count=0)

        if index > len(DISCOVERY_QUESTIONS):
            raise HTTPException(
                status_code=400,
                detail=f"question_index must be between 0 and {len(DISCOVERY_QUESTIONS)}.",
            )

        return self._build_discovery_response(answered_count=index)

    @staticmethod
    def _build_discovery_response(answered_count: int) -> DiscoveryStepResponse:
        is_complete = answered_count == len(DISCOVERY_QUESTIONS)
        current_index = None if answered_count == 0 else answered_count
        current_question = None if answered_count == 0 else DISCOVERY_QUESTIONS[answered_count - 1]
        next_index = None if is_complete else answered_count + 1
        next_question = None if is_complete else DISCOVERY_QUESTIONS[answered_count]
        return DiscoveryStepResponse(
            answered_count=answered_count,
            is_complete=is_complete,
            current_question_index=current_index,
            current_question=current_question,
            next_question_index=next_index,
            next_question=next_question,
        )

    async def generate_design_advice(self, payload: CopilotRequest) -> DesignAdviceResponse:
        project_idea = self._extract_project_idea(payload)
        grounded_prompt = self._build_grounded_user_prompt(payload.qa_context)
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
            f"{json.dumps([item.model_dump() for item in payload.qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(DesignAdviceResponse, data)

    async def generate_roadmap(self, payload: CopilotRequest) -> RoadmapResponse:
        project_idea = self._extract_project_idea(payload)
        grounded_prompt = self._build_grounded_user_prompt(payload.qa_context)
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
            f"{json.dumps([item.model_dump() for item in payload.qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(RoadmapResponse, data)

    async def generate_full_plan(self, payload: CopilotRequest) -> FullPlanResponse:
        design_advice = await self.generate_design_advice(payload)
        roadmap = await self.generate_roadmap(payload)
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

    @staticmethod
    def _extract_project_idea(payload: CopilotRequest) -> str:
        if payload.project_idea:
            return payload.project_idea
        if payload.qa_context:
            return payload.qa_context[0].answer
        raise HTTPException(status_code=400, detail="No project idea available in request context.")

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
