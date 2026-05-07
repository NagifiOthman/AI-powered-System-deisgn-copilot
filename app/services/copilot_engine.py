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
    DiscoveryQuestionsResponse,
    FullPlanResponse,
    RoadmapResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class CopilotEngine:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate_discovery_questions(self, payload: CopilotRequest) -> DiscoveryQuestionsResponse:
        schema_hint = {
            "questions": [
                "What scale (users/requests) do you expect in year one?",
                "What compliance/security requirements apply (SOC2, HIPAA, GDPR)?",
                "What MVP launch timeline are you targeting?",
            ]
        }
        user_prompt = (
            "Generate architecture-impacting discovery questions. "
            "Return JSON only matching this shape:\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Rules: 3 to 8 concise questions, no marketing questions.\n\n"
            "Project idea:\n"
            f"{payload.project_idea}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(DiscoveryQuestionsResponse, data)

    async def generate_design_advice(self, payload: CopilotRequest) -> DesignAdviceResponse:
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
            f"{payload.project_idea}\n\n"
            "Q/A context:\n"
            f"{json.dumps([item.model_dump() for item in payload.qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(DesignAdviceResponse, data)

    async def generate_roadmap(self, payload: CopilotRequest) -> RoadmapResponse:
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
            f"{payload.project_idea}\n\n"
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
