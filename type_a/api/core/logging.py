import logging
import sys
from app.core.config import settings

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log")
        ]
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    
    # Create logger for this application
    logger = logging.getLogger("rfp_agent")
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging() 