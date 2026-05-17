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
    RequirementsContextRequest,
    RequirementsContextResponse,
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
        self._requirements_context: dict[str, list[str]] | None = None

    async def get_discovery_step(self, payload: DiscoveryProgressRequest) -> DiscoveryStepResponse:
        if payload.restart:
            self._discovery_answers = []
            self._pending_question_index = None
            self._requirements_context = None

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
                    detail="Discovery is already complete. Use restart=true to start over. Or check the generated design advice and roadmap",
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
        validated_answer = await self._validate_discovery_answer(
            question_index=self._pending_question_index,
            question=expected_question,
            answer=payload.answer,
            prior_answers=self._discovery_answers,
        )
        self._discovery_answers.append(
            QAPair(question=expected_question, answer=validated_answer)
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

    async def set_requirements_context(
        self,
        payload: RequirementsContextRequest,
    ) -> RequirementsContextResponse:
        _, qa_context = self._require_completed_discovery_context()

        validation = await self._validate_requirements_context(
            functional_requirements=payload.functional_requirements,
            non_functional_requirements=payload.non_functional_requirements,
            discovery_context=qa_context,
        )

        self._requirements_context = {
            "functional_requirements": validation["functional_requirements"],
            "non_functional_requirements": validation["non_functional_requirements"],
        }

        return RequirementsContextResponse(
            is_valid=True,
            message="Requirements context accepted and stored.",
            functional_requirements=self._requirements_context["functional_requirements"],
            non_functional_requirements=self._requirements_context["non_functional_requirements"],
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

    def _require_ready_planning_context(self) -> tuple[str, list[QAPair], list[str], list[str]]:
        project_idea, qa_context = self._require_completed_discovery_context()
        if not self._requirements_context:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Submit functional and non-functional requirements before generating plans. "
                    "Call POST /api/v1/requirements-context first."
                ),
            )

        return (
            project_idea,
            qa_context,
            self._requirements_context["functional_requirements"],
            self._requirements_context["non_functional_requirements"],
        )

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
        project_idea, qa_context, functional_requirements, non_functional_requirements = (
            self._require_ready_planning_context()
        )
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
            "Validated functional requirements:\n"
            f"{json.dumps(functional_requirements, indent=2)}\n\n"
            "Validated non-functional requirements:\n"
            f"{json.dumps(non_functional_requirements, indent=2)}\n\n"
            "Q/A context:\n"
            f"{json.dumps([item.model_dump() for item in qa_context], indent=2)}"
        )
        data = await self._request_json(self._build_system_prompt(), user_prompt)
        return self._validate_model(DesignAdviceResponse, data)

    async def generate_roadmap(self) -> RoadmapResponse:
        project_idea, qa_context, functional_requirements, non_functional_requirements = (
            self._require_ready_planning_context()
        )
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
            "Validated functional requirements:\n"
            f"{json.dumps(functional_requirements, indent=2)}\n\n"
            "Validated non-functional requirements:\n"
            f"{json.dumps(non_functional_requirements, indent=2)}\n\n"
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
                grounded_kb = json.dumps(CURATED_KNOWLEDGE_BASE)
                return (
                        "You are an elite AI System Design Copilot for production-grade architecture. "
                        "Your outputs must be product-specific, constraint-aware, and implementation-ready, never generic. "
                        "\n\n"
                        "ANALYSIS MODE:\n"
                        "Infer and reason about: domain, business objective, user growth, latency/throughput, availability targets, "
                        "security/compliance, budget, team maturity, timeline, data sensitivity, integration dependencies, and risks. "
                        "State key assumptions when inputs are incomplete.\n\n"
                        "ARCHITECTURE EXPECTATIONS:\n"
                        "Evaluate and justify decisions across architecture style, service boundaries, data architecture, auth model, "
                        "caching, async processing, resilience, CI/CD, observability, automation, DR/backup, testing strategy, and AI safety. "
                        "Always include tradeoffs and operational impact.\n\n"
                        "DOMAIN ADAPTATION:\n"
                        "Tailor recommendations by domain signals (FinTech, HealthTech, E-commerce, SaaS, AI platforms, cybersecurity, IoT, social, enterprise internal tools). "
                        "Do not reuse the same stack patterns across unrelated contexts.\n\n"
                        "PRIORITIZATION:\n"
                        "Differentiate MVP-stage decisions from scale-stage evolution. Prefer pragmatic sequencing over theoretical perfection. "
                        "Recommend technologies only when justified by context and constraints.\n\n"
                        "OUTPUT RULES:\n"
                        "Return valid JSON only. No markdown, no code fences, no filler. Keep responses concise but technically deep, "
                        "with high-signal rationale and explicit risk callouts.\n\n"
                        f"GROUNDING KNOWLEDGE BASE: {grounded_kb}"
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

    async def _validate_discovery_answer(
        self,
        question_index: int,
        question: str,
        answer: str,
        prior_answers: list[QAPair],
    ) -> str:
        schema_hint = {
            "is_valid": True,
            "reason": "Short validation note",
            "normalized_answer": "Cleaned answer preserving user intent",
        }
        prior_context = [item.model_dump() for item in prior_answers]
        system_prompt = (
            "You validate discovery answers for a system-design copilot. "
            "Reject answers that are off-topic, nonsensical, or do not actually answer the asked question. "
            "Output JSON only."
        )
        user_prompt = (
            "Return JSON matching this shape:\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Validation rules:\n"
            "- is_valid=true only if the answer directly addresses the question.\n"
            "- Reject out-of-context, random, or irrelevant text.\n"
            "- Keep reason concise and actionable.\n"
            "- normalized_answer should be a trimmed, clarified version of a valid answer.\n\n"
            f"Question index: {question_index + 1}\n"
            f"Current question: {question}\n"
            f"User answer: {answer}\n"
            f"Prior discovery context: {json.dumps(prior_context)}"
        )
        data = await self._request_json(system_prompt, user_prompt)

        is_valid = bool(data.get("is_valid", False))
        reason = str(data.get("reason", "Answer is invalid for the current question.")).strip()
        normalized_answer = str(data.get("normalized_answer", "")).strip()

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid answer for discovery question {question_index + 1}: {reason}"
                ),
            )

        return normalized_answer if normalized_answer else answer.strip()

    async def _validate_requirements_context(
        self,
        functional_requirements: list[str],
        non_functional_requirements: list[str],
        discovery_context: list[QAPair],
    ) -> dict[str, list[str]]:
        schema_hint = {
            "is_valid": True,
            "reason": "Short validation note",
            "functional_requirements": ["First validated functional requirement"],
            "non_functional_requirements": ["First validated non-functional requirement"],
        }
        context = [item.model_dump() for item in discovery_context]
        system_prompt = (
            "You validate product functional and non-functional requirements for system design planning. "
            "Reject irrelevant, contradictory, or low-signal requirement sets. Output JSON only."
        )
        user_prompt = (
            "Return JSON matching this shape:\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Validation rules:\n"
            "- is_valid=true only if requirements are coherent and relevant to the discovery context.\n"
            "- Remove duplicates and rewrite vague items into concrete, concise requirements.\n"
            "- Keep functional requirements as capabilities and workflows.\n"
            "- Keep non-functional requirements as quality attributes, constraints, and SLO-like targets when possible.\n"
            "- Keep reason concise and actionable.\n\n"
            f"Discovery context: {json.dumps(context)}\n"
            f"Functional requirements input: {json.dumps(functional_requirements)}\n"
            f"Non-functional requirements input: {json.dumps(non_functional_requirements)}"
        )
        data = await self._request_json(system_prompt, user_prompt)

        is_valid = bool(data.get("is_valid", False))
        reason = str(data.get("reason", "Requirements are invalid for current project context.")).strip()
        validated_functional = [
            item.strip()
            for item in data.get("functional_requirements", [])
            if isinstance(item, str) and item.strip()
        ]
        validated_non_functional = [
            item.strip()
            for item in data.get("non_functional_requirements", [])
            if isinstance(item, str) and item.strip()
        ]

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid requirements context: {reason}",
            )

        if not validated_functional:
            validated_functional = [item.strip() for item in functional_requirements if item.strip()]
        if not validated_non_functional:
            validated_non_functional = [item.strip() for item in non_functional_requirements if item.strip()]

        return {
            "functional_requirements": self._dedupe_preserve_order(validated_functional),
            "non_functional_requirements": self._dedupe_preserve_order(validated_non_functional),
        }

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

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
