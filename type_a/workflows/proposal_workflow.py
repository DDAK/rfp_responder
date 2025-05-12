from typing import List, Dict, Any, Optional
from langgraph.graph import Graph, StateGraph
from app.core.logging import logger
from app.models.base import RFP, Proposal, Requirement, KnowledgeBaseEntry
from app.agents.document_processor import DocumentProcessor
from app.agents.requirement_extractor import RequirementExtractor
from app.agents.knowledge_base import KnowledgeBaseAgent
from app.agents.proposal_generator import ProposalGenerator

class ProposalWorkflow:
    """Main workflow orchestrator for the RFP proposal generation process."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.requirement_extractor = RequirementExtractor()
        self.knowledge_base = KnowledgeBaseAgent()
        self.proposal_generator = ProposalGenerator()
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the workflow graph using LangGraph."""
        # Define the state
        class WorkflowState:
            def __init__(self):
                self.rfp: Optional[RFP] = None
                self.requirements: List[Requirement] = []
                self.similar_proposals: List[Proposal] = []
                self.knowledge_base_entries: List[KnowledgeBaseEntry] = []
                self.proposal: Optional[Proposal] = None
                self.status: str = "initialized"
                self.errors: List[str] = []
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Define the nodes
        async def process_document(state: WorkflowState) -> WorkflowState:
            """Process the RFP document."""
            try:
                # Process the document
                doc_content = await self.document_processor.process_document(state.rfp.raw_text)
                state.rfp.raw_text = doc_content["content"]
                state.status = "document_processed"
            except Exception as e:
                state.errors.append(f"Error processing document: {str(e)}")
                state.status = "error"
            return state
        
        async def extract_requirements(state: WorkflowState) -> WorkflowState:
            """Extract requirements from the RFP."""
            try:
                state.requirements = await self.requirement_extractor.extract_requirements(state.rfp)
                state.status = "requirements_extracted"
            except Exception as e:
                state.errors.append(f"Error extracting requirements: {str(e)}")
                state.status = "error"
            return state
        
        async def find_similar_proposals(state: WorkflowState) -> WorkflowState:
            """Find similar proposals from the knowledge base."""
            try:
                state.similar_proposals = await self.knowledge_base.find_similar_proposals(state.rfp)
                state.status = "similar_proposals_found"
            except Exception as e:
                state.errors.append(f"Error finding similar proposals: {str(e)}")
                state.status = "error"
            return state
        
        async def search_knowledge_base(state: WorkflowState) -> WorkflowState:
            """Search the knowledge base for relevant content."""
            try:
                # Search for each requirement
                for req in state.requirements:
                    entries = await self.knowledge_base.search_knowledge_base(req.text)
                    state.knowledge_base_entries.extend(entries)
                state.status = "knowledge_base_searched"
            except Exception as e:
                state.errors.append(f"Error searching knowledge base: {str(e)}")
                state.status = "error"
            return state
        
        async def generate_proposal(state: WorkflowState) -> WorkflowState:
            """Generate the proposal."""
            try:
                state.proposal = await self.proposal_generator.generate_proposal(
                    state.rfp,
                    state.similar_proposals,
                    state.knowledge_base_entries
                )
                state.status = "proposal_generated"
            except Exception as e:
                state.errors.append(f"Error generating proposal: {str(e)}")
                state.status = "error"
            return state
        
        # Add nodes to the graph
        workflow.add_node("process_document", process_document)
        workflow.add_node("extract_requirements", extract_requirements)
        workflow.add_node("find_similar_proposals", find_similar_proposals)
        workflow.add_node("search_knowledge_base", search_knowledge_base)
        workflow.add_node("generate_proposal", generate_proposal)
        
        # Define the edges
        workflow.add_edge("process_document", "extract_requirements")
        workflow.add_edge("extract_requirements", "find_similar_proposals")
        workflow.add_edge("find_similar_proposals", "search_knowledge_base")
        workflow.add_edge("search_knowledge_base", "generate_proposal")
        
        # Set the entry point
        workflow.set_entry_point("process_document")
        
        return workflow.compile()
    
    async def run_workflow(self, rfp: RFP) -> Dict[str, Any]:
        """Run the complete proposal generation workflow."""
        logger.info(f"Starting proposal generation workflow for RFP: {rfp.id}")
        
        try:
            # Initialize the state
            initial_state = self.workflow.State()
            initial_state.rfp = rfp
            
            # Run the workflow
            final_state = await self.workflow.arun(initial_state)
            
            # Check for errors
            if final_state.errors:
                logger.error(f"Workflow completed with errors: {final_state.errors}")
                return {
                    "status": "error",
                    "errors": final_state.errors,
                    "proposal": None
                }
            
            # Return the results
            return {
                "status": "success",
                "proposal": final_state.proposal,
                "requirements": final_state.requirements,
                "similar_proposals": final_state.similar_proposals,
                "knowledge_base_entries": final_state.knowledge_base_entries
            }
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            return {
                "status": "error",
                "errors": [str(e)],
                "proposal": None
            }
    
    async def improve_proposal(self, proposal: Proposal, feedback: Dict[str, str]) -> Proposal:
        """Improve the proposal based on feedback."""
        logger.info(f"Improving proposal: {proposal.id}")
        
        try:
            # Improve each section based on feedback
            for section_name, section_feedback in feedback.items():
                if section_name in proposal.sections:
                    improved_content = await self.proposal_generator.improve_section(
                        proposal.sections[section_name],
                        section_feedback
                    )
                    proposal.sections[section_name] = improved_content
            
            # Update proposal status
            proposal.status = "revised"
            
            return proposal
            
        except Exception as e:
            logger.error(f"Error improving proposal: {str(e)}")
            raise
    
    def close(self):
        """Clean up resources."""
        self.knowledge_base.close() 