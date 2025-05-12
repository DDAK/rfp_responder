from fastapi import APIRouter
from app.api.v1.endpoints import proposals

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    proposals.router,
    prefix="/proposals",
    tags=["proposals"]
) 