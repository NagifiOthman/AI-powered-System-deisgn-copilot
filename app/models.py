from typing import Literal

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class CopilotRequest(BaseModel):
    project_idea: str = Field(
        min_length=10,
        description="User project idea or product description.",
    )
    qa_context: list[QAPair] = Field(
        default_factory=list,
        description="Optional answers to follow-up discovery questions.",
    )


class DiscoveryQuestionsResponse(BaseModel):
    questions: list[str] = Field(
        description="Short, high-signal discovery questions to refine architecture recommendations.",
        min_length=3,
        max_length=8,
    )


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
