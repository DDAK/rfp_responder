from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.models.base import RFP, Proposal, Requirement, KnowledgeBaseEntry
from app.workflows.proposal_workflow import ProposalWorkflow
from app.core.logging import logger
import uuid
import json

router = APIRouter()

# Initialize workflow
workflow = ProposalWorkflow()

@router.post("/rfp/upload")
async def upload_rfp(
    file: UploadFile = File(...),
    title: str = None,
    agency: str = None,
    solicitation_number: str = None
) -> Dict[str, Any]:
    """Upload and process an RFP document."""
    try:
        # Save the uploaded file
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create RFP object
        rfp = RFP(
            id=str(uuid.uuid4()),
            title=title or file.filename,
            agency=agency or "Unknown",
            solicitation_number=solicitation_number or "Unknown",
            raw_text=content.decode(),
            document_type=file.content_type.split("/")[-1]
        )
        
        # Process the RFP
        result = await workflow.run_workflow(rfp)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["errors"])
        
        return {
            "status": "success",
            "rfp_id": rfp.id,
            "proposal": result["proposal"]
        }
        
    except Exception as e:
        logger.error(f"Error processing RFP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rfp/{rfp_id}")
async def get_rfp(rfp_id: str) -> RFP:
    """Get RFP details."""
    # TODO: Implement RFP retrieval from database
    raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/proposal/{proposal_id}")
async def get_proposal(proposal_id: str) -> Proposal:
    """Get proposal details."""
    # TODO: Implement proposal retrieval from database
    raise HTTPException(status_code=501, detail="Not implemented")

@router.post("/proposal/{proposal_id}/improve")
async def improve_proposal(
    proposal_id: str,
    feedback: Dict[str, str]
) -> Proposal:
    """Improve a proposal based on feedback."""
    try:
        # TODO: Get proposal from database
        proposal = None  # Replace with actual proposal retrieval
        
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        # Improve the proposal
        improved_proposal = await workflow.improve_proposal(proposal, feedback)
        
        return improved_proposal
        
    except Exception as e:
        logger.error(f"Error improving proposal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-base/entry")
async def add_knowledge_entry(entry: KnowledgeBaseEntry) -> Dict[str, Any]:
    """Add a new entry to the knowledge base."""
    try:
        # TODO: Implement knowledge base entry addition
        return {"status": "success", "entry_id": entry.id}
        
    except Exception as e:
        logger.error(f"Error adding knowledge base entry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/search")
async def search_knowledge_base(
    query: str,
    category: str = None
) -> List[KnowledgeBaseEntry]:
    """Search the knowledge base."""
    try:
        # TODO: Implement knowledge base search
        return []
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/proposal/{proposal_id}/compliance")
async def get_proposal_compliance(proposal_id: str) -> Dict[str, Any]:
    """Get compliance analysis for a proposal."""
    try:
        # TODO: Implement compliance analysis
        return {"status": "success", "compliance": {}}
        
    except Exception as e:
        logger.error(f"Error analyzing compliance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 