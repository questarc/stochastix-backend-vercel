from __future__ import annotations

from fastapi import APIRouter

from ..methodology import METHODOLOGY_SECTIONS
from ..schemas import MethodologySection

router = APIRouter(tags=["methodology"])


@router.get("/methodology", response_model=list[MethodologySection])
def get_methodology():
    return METHODOLOGY_SECTIONS
