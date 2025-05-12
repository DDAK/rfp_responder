from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncpg
import os
import json
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Get database connection settings from environment variables
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag")
DB_USER = os.getenv("DB_USER", "demo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "demo1234")

app = FastAPI(title="Job Queue API")

# Database connection pool
pool = None

class JobRequest(BaseModel):
    """Model for job request"""
    job_type: str
    parameters: Dict[str, Any]

class JobResponse(BaseModel):
    """Model for job response"""
    job_id: int
    status: str
    created_at: datetime

@app.on_event("startup")
async def startup():
    """Create database connection pool on startup"""
    global pool
    try:
        logger.info("Creating database connection pool...")
        pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            min_size=5,
            max_size=20
        )
        logger.info("Database connection pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create database connection pool: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool on shutdown"""
    global pool
    if pool:
        await pool.close()
        logger.info("Database connection pool closed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Job Queue API is running"}

@app.post("/jobs", response_model=JobResponse)
async def create_job(job_request: JobRequest):
    """Create a new job"""
    async with pool.acquire() as conn:
        try:
            # Insert job into queue
            job_data = {
                "job_type": job_request.job_type,
                "parameters": job_request.parameters
            }
            
            query = """
                INSERT INTO job_queue (status, job_data)
                VALUES ('pending', $1)
                RETURNING job_id, status, created_at
            """
            row = await conn.fetchrow(query, json.dumps(job_data))
            
            logger.info(f"Job created with ID: {row['job_id']}")
            
            return JobResponse(
                job_id=row["job_id"],
                status=row["status"],
                created_at=row["created_at"]
            )
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job(job_id: int):
    """Get job details by ID"""
    async with pool.acquire() as conn:
        try:
            query = """
                SELECT job_id, status, created_at, started_at, completed_at, 
                       job_data, result, worker_id, error_message
                FROM job_queue
                WHERE job_id = $1
            """
            row = await conn.fetchrow(query, job_id)
            
            if not row:
                raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
            
            # Convert record to dict and handle datetime objects
            job_dict = dict(row)
            for key, value in job_dict.items():
                if isinstance(value, datetime):
                    job_dict[key] = value.isoformat()
            
            return job_dict
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Failed to fetch job: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch job: {str(e)}")

@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100, offset: int = 0):
    """List jobs with optional status filter"""
    async with pool.acquire() as conn:
        try:
            if status:
                query = """
                    SELECT job_id, status, created_at, started_at, completed_at
                    FROM job_queue
                    WHERE status = $1
                    ORDER BY job_id DESC
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, status, limit, offset)
            else:
                query = """
                    SELECT job_id, status, created_at, started_at, completed_at
                    FROM job_queue
                    ORDER BY job_id DESC
                    LIMIT $1 OFFSET $2
                """
                rows = await conn.fetch(query, limit, offset)
            
            # Convert records to list of dicts
            jobs = []
            for row in rows:
                job_dict = dict(row)
                for key, value in job_dict.items():
                    if isinstance(value, datetime):
                        job_dict[key] = value.isoformat()
                jobs.append(job_dict)
            
            return {"jobs": jobs, "count": len(jobs)}
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get queue statistics"""
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow("SELECT * FROM queue_stats")
            return dict(row)
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)