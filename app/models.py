from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QAPair(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class CopilotRequest(BaseModel):
    project_idea: str | None = Field(
        default=None,
        min_length=10,
        description="User project idea or product description.",
    )
    qa_context: list[QAPair] = Field(
        default_factory=list,
        description="Optional answers to follow-up discovery questions.",
    )

    @model_validator(mode="after")
    def validate_minimum_context(self):
        if self.project_idea:
            return self
        if not self.qa_context:
            raise ValueError("Provide either project_idea or at least one qa_context answer.")
        return self


class DiscoveryProgressRequest(BaseModel):
    question_index: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Question index being answered. Use 0 to fetch the first question.",
    )
    answer: str | None = Field(
        default=None,
        description="Answer for the specified question index (required for index 1..5).",
    )

    @model_validator(mode="after")
    def validate_discovery_step(self):
        if self.question_index == 0 and self.answer:
            raise ValueError("Do not provide an answer when question_index is 0.")
        if self.question_index > 0 and (self.answer is None or not self.answer.strip()):
            raise ValueError("Provide a non-empty answer for question_index 1..5.")
        return self


class DiscoveryStepResponse(BaseModel):
    total_questions: int = 5
    answered_count: int
    is_complete: bool
    current_question_index: int | None
    current_question: str | None
    next_question_index: int | None
    next_question: str | None


class TechRecommendation(BaseModel):
    component: str
    recommendation: str
    rationale: str


class SecurityDevOpsChecklistItem(BaseModel):
    area: Literal["security", "devops"]
    item: str
    priority: Literal["high", "medium", "low"]


class DesignAdviceResponse(BaseModel):
    recommended_tech_stack: list[TechRecommendation]
    security_and_devops_checklist: list[SecurityDevOpsChecklistItem]


class FeatureBacklogItem(BaseModel):
    name: str
    description: str
    effort: Literal["small", "medium", "large"]
    impact: Literal["low", "medium", "high"]
    estimate_range: str = Field(description="Example: 2-4 days, 1-2 weeks")


class DependencyEdge(BaseModel):
    feature: str
    depends_on: str
    reason: str


class RoadmapResponse(BaseModel):
    mvp_core_features: list[FeatureBacklogItem]
    post_mvp_differentiators: list[FeatureBacklogItem]
    dependency_graph: list[DependencyEdge]


class FullPlanResponse(BaseModel):
    design_advice: DesignAdviceResponse
    roadmap: RoadmapResponse
