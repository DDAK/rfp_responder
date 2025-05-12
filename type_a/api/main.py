from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .v1.api import api_router
from .core.logging import setup_logging

app = FastAPI(
    title="RFP Agent System",
    description="Intelligent RFP Response Generation System",
    version="1.0.0",
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to RFP Agent System"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 