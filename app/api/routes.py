from fastapi import APIRouter

from app.models import (
    DesignAdviceResponse,
    DiscoveryProgressRequest,
    DiscoveryStepResponse,
    FullPlanResponse,
    RequirementsContextRequest,
    RequirementsContextResponse,
    RoadmapResponse,
)
from app.services.copilot_engine import copilot_engine

router = APIRouter(tags=["Copilot"])


@router.post("/discovery-questions", response_model=DiscoveryStepResponse)
async def discovery_questions(payload: DiscoveryProgressRequest) -> DiscoveryStepResponse:
    return await copilot_engine.get_discovery_step(payload)


@router.post("/requirements-context", response_model=RequirementsContextResponse)
async def requirements_context(payload: RequirementsContextRequest) -> RequirementsContextResponse:
    return await copilot_engine.set_requirements_context(payload)


@router.post("/design-advice", response_model=DesignAdviceResponse)
async def design_advice() -> DesignAdviceResponse:
    return await copilot_engine.generate_design_advice()


@router.post("/roadmap", response_model=RoadmapResponse)
async def roadmap() -> RoadmapResponse:
    return await copilot_engine.generate_roadmap()


@router.post("/full-plan", response_model=FullPlanResponse)
async def full_plan() -> FullPlanResponse:
    return await copilot_engine.generate_full_plan()
