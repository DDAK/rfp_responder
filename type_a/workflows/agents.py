import os
import json
import asyncio
from typing import List, Any, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages

import langgraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger
from lightrag.kg.shared_storage import initialize_pipeline_status

os.makedirs("./rag_storage", exist_ok=True)

# Set environment variables
os.environ.setdefault("NEO4J_URI", os.getenv("NEO4J_URI", "neo4j://localhost:7687"))
os.environ.setdefault("NEO4J_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j"))
os.environ.setdefault("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "mypassword"))
os.environ.setdefault("MONGO_URI", os.getenv("MONGO_URI", "mongodb://mongoUser:mongoPass@localhost:27017/"))
os.environ.setdefault("MONGO_DATABASE", os.getenv("MONGO_DATABASE", "LightRAG"))
os.environ.setdefault("POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost"))
os.environ.setdefault("POSTGRES_PORT", os.getenv("POSTGRES_PORT", "15432"))
os.environ.setdefault("POSTGRES_USER", os.getenv("POSTGRES_USER", "demo"))
os.environ.setdefault("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "demo1234"))
os.environ.setdefault("POSTGRES_DATABASE", os.getenv("POSTGRES_DATABASE", "rag"))

# Create a state type for the graph
class GraphState(MessagesState):
    """State for the graph."""
    messages: List[Any] = Field(default_factory=list)
    next: Literal["research_agent", "writer_agent", "supervisor", "end"] = "supervisor"
    should_end: bool = False
    rag: Optional[LightRAG] = None

class LightRAGRetriever(BaseRetriever, BaseModel):
    """
    A LangChain Retriever that delegates to LightRAG's query() method.
    """
    mode: str = "hybrid"
    rag: Optional[LightRAG] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.rag:
            # Create LightRAG instance synchronously
            self.rag = LightRAG(
                working_dir="./rag_storage",
                llm_model_func=ollama_model_complete,
                llm_model_name="qwen2.5:latest",
                llm_model_max_async=4,
                llm_model_max_token_size=32768,
                llm_model_kwargs={
                    "host": "http://127.0.0.1:11434",
                    "options": {"num_ctx": 32768},
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=4096,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts=texts, 
                        embed_model="bge-m3:latest", 
                        host="http://127.0.0.1:11434"
                    ),
                ),
                kv_storage="MongoKVStorage",
                graph_storage="Neo4JStorage",
                vector_storage="PGVectorStorage",
                doc_status_storage="PGDocStatusStorage",
            )

    async def initialize(self):
        """Initialize the retriever asynchronously."""
        if self.rag:
            await self.rag.initialize_storages()
            await initialize_pipeline_status()

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from LightRAG."""
        try:
            if not self.rag:
                raise ValueError("LightRAG instance not initialized")
            
            # Ensure initialization
            await self.initialize()
            
            param = QueryParam(mode=self.mode)
            query_prompt = "Query:{}, Instructions:{}".format(query, "When you answer, return strictly valid JSON:  \
                        [ \
                        { \"text\": \"<passage text>\", \"meta\": { … } }, \
                        … \
                        ] \
                        ")
            response = await self.rag.aquery(query_prompt, param=param)
            docs = [Document(page_content=response, metadata={})]
            return docs
        except Exception as e:
            logger.error(f"Error in aget_relevant_documents: {e}")
            return []

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous wrapper for aget_relevant_documents."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.aget_relevant_documents(query))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {e}")
            return []

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.rag:
                await self.rag.finalize_storages()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'rag') and self.rag:
                # Create a new event loop for cleanup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.cleanup())
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")

# Create the graph
workflow = langgraph.graph.StateGraph(GraphState)

# Create retriever with shared LightRAG instance
graph_rag_retriever = LightRAGRetriever(mode="hybrid")

# Initialize the retriever
async def initialize_retriever():
    await graph_rag_retriever.initialize()

# Run initialization
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(initialize_retriever())
finally:
    loop.close()

# expose it as a LangChain tool
graphrag_tool = create_retriever_tool(
    retriever=graph_rag_retriever,
    name="graph_rag_qa",
    description=(
        "Use this tool to get RFP requirements and sections by querying our knowledge graph via LightRAG. "
        "Supports hybrid (graph + vector) search out of the box."
    ),
)

llm = init_chat_model("qwen2.5:latest", model_provider="ollama", temperature=0)

#--- Research Agent ---------------------------------
research_agent = create_react_agent(
    llm,
    tools=[graphrag_tool],
    prompt="""You are a research agent specialized in analyzing RFP documents and requirements.
Your primary responsibilities are:
1. Search and retrieve relevant information from the RFP knowledge base
2. Identify key requirements, specifications, and evaluation criteria
3. Find supporting evidence and documentation
4. Analyze relationships between requirements and sections

When using the graph_rag_qa tool:
- Use specific, targeted queries to find relevant information
- Look for both explicit requirements and implicit dependencies
- Consider compliance criteria and evaluation factors
- Gather supporting evidence and documentation

Always structure your findings clearly, highlighting:
- Critical requirements
- Dependencies and relationships
- Compliance criteria
- Supporting evidence

Remember to verify the relevance and completeness of your findings.""",
    name="research_agent"
)

#--- Writer Agent ---------------------------------
writer_agent = create_react_agent(
    llm,
    tools=[graphrag_tool],
    prompt="""You are a writer agent specialized in creating RFP responses.
Your primary responsibilities are:
1. Draft clear and compelling responses to RFP requirements
2. Ensure all requirements are addressed comprehensively
3. Incorporate supporting evidence and documentation
4. Maintain consistent formatting and professional tone

When writing responses:
- Address each requirement directly and completely
- Use clear, professional language
- Support claims with evidence
- Follow RFP formatting guidelines
- Highlight key strengths and differentiators

Structure your responses to:
- Clearly address each requirement
- Include relevant evidence and examples
- Maintain logical flow and organization
- Use appropriate technical terminology

Remember to:
- Stay focused on the specific requirements
- Use evidence to support your claims
- Maintain a professional and persuasive tone
- Follow any specified formatting guidelines""",
    name="writer_agent"
)

#--- supervisor agent--------------------------------

system_prompt = """
You are an expert RFP analysis agent designed to interact with a comprehensive RFP Knowledge Graph system.
Your primary goal is to help users find relevant information about Securing Artificial Intelligence for Battlefield Effective Robustness (SABER) RFP documents, requirements, and related materials.

When given a question:
1. Use the graph_rag_qa tool to search across:
   - Knowledge Graph (for structured relationships and requirements)
   - Vector Database (for semantic similarity and context)
   - KV Store (for document metadata and references)

2. Always consider:
   - Direct requirements and specifications
   - Related sections and dependencies
   - Supporting evidence and documentation
   - Compliance criteria and evaluation factors

3. For each query:
   - Focus on the most relevant information first
   - Include context and relationships when available
   - Highlight key requirements and their dependencies
   - Note any compliance or evaluation criteria

4. When presenting results:
   - Structure the information clearly
   - Highlight critical requirements
   - Include relevant context and relationships
   - Note any dependencies or prerequisites

Remember to:
- Always verify the relevance of retrieved information
- Consider both explicit requirements and implicit dependencies
- Look for related sections and supporting documentation
- Highlight any compliance or evaluation criteria

The system uses hybrid search (graph + vector) to find the most relevant information. Use this capability to get comprehensive results that combine structured relationships with semantic understanding.
"""

supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, writer_agent],
    tools=[graphrag_tool],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a writer agent. Assign writing-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself.\n"
        "You have access to the graph_rag_qa tool for querying the knowledge base."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile(name="supervisor")


# Define the end node
def end(state: GraphState) -> GraphState:
    """End the conversation."""
    return state
# Add nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("research_agent", research_agent)
workflow.add_node("writer_agent", writer_agent)
workflow.add_node("end", end)

# Define the routing logic
def router(state: GraphState | dict) -> Literal["research_agent", "writer_agent", "supervisor", "end"]:
    """Route to the next agent based on the state."""
    # Handle state conversion without type checking
    if isinstance(state, dict):
        state = GraphState(
            messages=state.get("messages", []),
            next=state.get("next", "supervisor"),
            should_end=state.get("should_end", False),
            rag=state.get("rag")
        )
    
    # Access state attributes safely
    should_end = getattr(state, "should_end", False)
    next_step = getattr(state, "next", "research_agent")
    
    if should_end:
        return "end"
    
    # Ensure we only return valid next steps
    if next_step not in ["research_agent", "writer_agent", "end"]:
        return "research_agent"
        
    return next_step

# Add edges with conditional routing
workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "end": "end"
    }
)

# Add edges back to supervisor
workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")

# Set entry point
workflow.set_entry_point("supervisor")

# Compile the graph
app = workflow.compile()

# Main execution
def run_graph():
    initial_state = GraphState(
        messages=[{"role": "user", "content": system_prompt}],
        should_end=False,
    )
    # Create a new event loop for the main execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        resp = loop.run_until_complete(app.ainvoke(initial_state, config={"recursion_limit": 50}))
        print(resp)
    finally:
        loop.close()

# Run the graph
if __name__ == "__main__":
    run_graph()

# inputs = {"messages": [{"role": "user", "content": f"{system_prompt}"}]}
# resp = supervisor.invoke(inputs)
# # pretty_print_messages(resp, last_message=True)
# print(resp)
# # final_message_history = chunk["supervisor"]["messages"]
# #----------------------------------------------------

