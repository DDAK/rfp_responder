import os
from typing import List, Any, Literal, Annotated, Dict, TypedDict, Union
from pydantic import Field, BaseModel

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph_supervisor import create_supervisor
from langgraph.types import Command, Send
from workflows.tools import graphrag_tool, graph_rag_retriever, TestGraphRAGRetriever, test_graph_rag_retriever

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")
        
def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        # print(f"DEBUG: handoff_tool: Agent name: {agent_name}, Task description: {task_description}")
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# Set up environment variables
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Please set it using: export GROQ_API_KEY='your-api-key'"
    )


# Initialize LLM with error handling
try:
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Error initializing Groq LLM: {e}")
    print("Falling back to Ollama...")
    llm = init_chat_model("qwen2.5:latest", model_provider="ollama", temperature=0)


# Create a debug-enabled retriever that wraps the original
class DebugTestGraphRAGRetriever(TestGraphRAGRetriever):
    """A retriever that adds debugging to the TestGraphRAGRetriever."""
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents with debugging."""
        print(f"\n>>> DEBUG: TestGraphRAGRetriever queried with: {query}")
        try:
            docs = super()._get_relevant_documents(query)
            print(f">>> DEBUG: Retrieved {len(docs)} documents")
            
            # Add a special document at the beginning to force the agent to summarize
            docs.insert(0, Document(
                page_content=f"IMPORTANT: You have successfully used the tool! Now summarize the following information about the SABER RFP:",
                metadata={"source": "system"}
            ))
            
            return docs
        except Exception as e:
            print(f">>> DEBUG: Error in retriever: {e}")
            return [Document(page_content=f"Error retrieving documents: {e}. Please try with a different query.", metadata={})]

# Create a debug-enabled retriever and tool
debug_retriever = DebugTestGraphRAGRetriever()
debug_test_graphrag_tool = create_retriever_tool(
    retriever=debug_retriever,
    name="test_graph_rag_qa",
    description="Use this tool to query our RFP knowledge graph. Input should be a specific query about RFP requirements.",
)

# Research agent setup with forced tool usage

research_agent = create_react_agent(
    llm,
    tools=[debug_test_graphrag_tool],
    prompt=(
        "You are a research agent with access to a tool called test_graph_rag_qa.\n\n"
        "INSTRUCTIONS:\n"
        "- You MUST use the test_graph_rag_qa tool for ALL research tasks\n"
        "- ALWAYS start by using the tool with a clear query\n"
        "- NEVER respond without first using the tool\n"
        "- After using the tool, summarize the findings and return them\n"
        "- Be efficient and complete your research quickly\n"
        "- Provide concise, relevant information\n\n"
        "You are a research agent with access to a tool called test_graph_rag_qa.\n\n"
        "INSTRUCTIONS:\n"
        "- You MUST use the test_graph_rag_qa tool for ALL research tasks\n"
        "- ALWAYS start by using the tool with a clear query\n"
        "- NEVER respond without first using the tool\n"
        "- After using the tool, summarize the findings and return them\n"
        "- Be efficient and complete your research quickly\n"
        "- Provide concise, relevant information\n\n"
        "HOW TO USE THE TOOL:\n"
        "1. ALWAYS use the tool with a query parameter\n"
        "2. Format your query like: test_graph_rag_qa(query=\"What are the requirements for SABER RFP?\")\n"
        "3. If you get an error, try rephrasing your query\n\n"
        "EXAMPLE TOOL USAGE:\n"
        "test_graph_rag_qa(query=\"What are the key requirements for the SABER RFP?\")\n"
        "test_graph_rag_qa(query=\"List the technical specifications in the SABER RFP\")\n"
        "test_graph_rag_qa(query=\"What are the evaluation criteria for the SABER RFP?\")\n\n"
        "IMPORTANT: You MUST use the tool before responding. Do not skip using the tool."
    ),
    name="research_agent"
)

# Writer agent setup
writer_agent = create_react_agent(
    llm,
    tools=[],
    prompt=(
        "You are a writer agent specialized in creating RFP responses.\n"
        "Your job is to write a proposal based on the research provided.\n\n"
        "INSTRUCTIONS:\n"
        "- If you receive complete research, create a comprehensive response\n"
        "- If you receive partial or incomplete research, do your best with what you have\n"
        "- If no research was provided, create a generic response and mention that more information would be helpful\n"
        "- Structure your response clearly with sections and bullet points\n"
        "- Be professional and concise\n\n"
        "Your response should include:\n"
        "1. A brief summary of the RFP requirements (if available)\n"
        "2. A proposed approach to meeting these requirements\n"
        "3. Any clarification questions if information is missing\n\n"
        "Even with limited information, provide the best possible response."
    ),
    name="writer_agent"
)

# Create supervisor with handoff tools
assign_to_research_agent_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent."
)

assign_to_writer_agent_with_description = create_task_description_handoff_tool(
    agent_name="writer_agent",
    description="Assign task to a writer agent."
)

# Define supervisor
supervisor_prompt = """You are a supervisor coordinating between a research agent and writer agent.
Your role is to:
1. Assign research tasks to the research agent
2. Send the research results to the writer agent
3. Provide final answers to the user

IMPORTANT: You must use EXACTLY these phrases to route to agents:
- To route to the research agent, say EXACTLY: "I'll ask the research agent to help with this."
- To route to the writer agent, say EXACTLY: "I'll ask the writer agent to create a response based on this research."

Workflow:
1. When a user asks a question, respond with "I'll ask the research agent to help with this."
2. After receiving research results, respond with "I'll ask the writer agent to create a response based on this research."
3. After receiving the written response, provide it to the user and end with "I believe we have completed the task. Let's end here."

IMPORTANT RULES:
- If the research agent returns empty results or doesn't use the tool, wait for one more response.
- If the research agent fails to use the tool twice, proceed to the writer agent anyway with whatever information you have.
- If the research agent keeps returning empty responses, move on to the writer agent.
- DO NOT get stuck in a loop with the research agent.

DO NOT skip these exact phrases, as they are required for the system to work correctly.

IMPORTANT: You are limited to a maximum of 15 turns. Please be efficient and complete the task quickly.
If you notice the conversation is going in circles, end it with "I believe we have completed the task. Let's end here."
"""

# Define proper state class
class GraphState(MessagesState):
    """The state type for the graph."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    next: str = Field(default="supervisor")  # Default next node to route to
    turn_count: int = Field(default=0)  # Track number of turns
    max_turns: int = Field(default=15)  # Maximum number of turns before forced termination
    research_attempts: int = Field(default=0)  # Track research attempts
    
    class Config:
        arbitrary_types_allowed = True

# Create a simple supervisor without complex tool binding
supervisor = create_react_agent(
    llm,
    tools=[],  # No tools for simplicity
    prompt=supervisor_prompt,
    name="supervisor"
)

# Build graph workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("research_agent", research_agent)
workflow.add_node("writer_agent", writer_agent)

# Define routing for each node type
def route_from_supervisor(state):
    """Route based on messages and keywords"""
    # Debug information
    print("DEBUG: Routing from supervisor")
    print(f"DEBUG: State type: {type(state)}")
    
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0) + 1
        max_turns = state.get("max_turns", 15)
        research_attempts = state.get("research_attempts", 0)
        print("DEBUG: State is dict")
    else:
        # Assuming it's a GraphState object
        messages = getattr(state, "messages", [])
        turn_count = getattr(state, "turn_count", 0) + 1
        max_turns = getattr(state, "max_turns", 15)
        research_attempts = getattr(state, "research_attempts", 0)
        print("DEBUG: State is GraphState")
    
    # Update turn count
    if isinstance(state, dict):
        state["turn_count"] = turn_count
    else:
        state.turn_count = turn_count
    
    print(f"DEBUG: Turn count: {turn_count}/{max_turns}")
    print(f"DEBUG: Messages count: {len(messages)}")
    
    # Force termination if max turns reached
    if turn_count >= max_turns:
        print("DEBUG: Max turns reached, forcing END")
        return END
    
    if not messages:
        print("DEBUG: No messages, routing to supervisor")
        return "supervisor"
    
    # Get the last message
    last_msg = messages[-1] if messages else None
    print(f"DEBUG: Last message type: {type(last_msg)}")
    
    # Extract content safely based on the message type
    content = ""
    if last_msg is not None:
        if hasattr(last_msg, "content"):
            # Handle AIMessage or other Message objects
            content = last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            # Handle dictionary messages
            content = last_msg["content"]
    
    print(f"DEBUG: Last message content: {content[:50]}...")
    
    # Check if we're in a loop between supervisor and research agent
    if "I'll ask the research agent" in content:
        # Track research attempts
        if isinstance(state, dict):
            research_attempts = state.get("research_attempts", 0) + 1
            state["research_attempts"] = research_attempts
        else:
            research_attempts = getattr(state, "research_attempts", 0) + 1
            state.research_attempts = research_attempts
        
        print(f"DEBUG: Research attempts: {research_attempts}")
        
        # If we've tried the research agent too many times, go to writer instead
        if research_attempts >= 3:
            print("DEBUG: Too many research attempts, forcing writer_agent")
            # Reset research attempts
            if isinstance(state, dict):
                state["research_attempts"] = 0
            else:
                state.research_attempts = 0
            return "writer_agent"
        
        print("DEBUG: Routing to research_agent")
        return "research_agent"
    
    if "I'll ask the writer agent" in content:
        print("DEBUG: Routing to writer_agent")
        # Reset research attempts
        if isinstance(state, dict):
            state["research_attempts"] = 0
        else:
            state.research_attempts = 0
        return "writer_agent"
    
    if "I believe we have completed the task" in content:
        print("DEBUG: Routing to END")
        return END
    
    print("DEBUG: Default routing to supervisor")
    return "supervisor"

# Add edges with proper routing
workflow.add_edge(START, "supervisor")

# Use conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "supervisor": "supervisor",
        END: END
    }
)

# Always route back to supervisor after agent completes
workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")

# Compile the app
app = workflow.compile()

# Main execution
if __name__ == "__main__":
    # Choose whether to stream or invoke
    use_streaming = True  # Set to False to use invoke instead
    
    result = None
    try:
        # Use a more specific query that encourages tool usage
        query = "What are the specific requirements and evaluation criteria for the SABER RFP? Please use the research tool to find this information."
        print(f"Creating initial state with query: {query}")
        
        # Create initial state with proper message objects and turn tracking
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "turn_count": 0,
            "max_turns": 15,
            "research_attempts": 0
        }
        
        # Configure execution with higher recursion limit and timeout
        config = {
            "recursion_limit": 50,  # Increased from default 25
            "timeout": 300,  # 5 minute timeout
            "interrupt_before_recursion_limit": True  # Get partial results before hitting limit
        }
        
        print("Starting workflow with query:", query)
        print("="*50)
        
        if use_streaming:
            # Stream results in real-time
            print("Streaming workflow results...")
            for chunk in app.stream(initial_state, config=config):
                try:
                    pretty_print_messages(chunk)
                except Exception as e:
                    print(f"Error printing chunk: {e}")
                    print(chunk)
            
            # Get the final state after streaming
            try:
                from langgraph.pregel import get_last_state
                result = get_last_state()
            except Exception as e:
                print(f"Failed to get final state: {e}")
        else:
            # Use invoke for batch processing
            print("Invoking workflow...")
            result = app.invoke(initial_state, config=config)
        
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        
        # Check if it's a recursion limit error
        if "Recursion limit" in str(e):
            print("Recursion limit reached, attempting to get partial results...")
            
            # Try to get the last state from the graph
            try:
                from langgraph.pregel import get_last_state
                last_state = get_last_state()
                if last_state:
                    print("Found partial results!")
                    result = last_state
            except Exception as inner_e:
                print(f"Failed to get partial results: {inner_e}")
        else:
            traceback.print_exc()
    
    finally:
        print("\nFinal result:")
        print("="*50)
        
        # Use pretty_print_messages to display final messages
        if result and isinstance(result, dict) and "messages" in result:
            # Format the result for pretty_print_messages
            formatted_update = {"final_result": {"messages": result["messages"]}}
            try:
                pretty_print_messages(formatted_update)
            except Exception as e:
                print(f"Error printing messages: {e}")
                # Fallback to basic printing
                messages = result["messages"]
                print(f"Found {len(messages)} messages in result")
                for msg in messages:
                    print(f"\n{msg}")
        else:
            print("No result or unexpected result format")
            print(f"Result type: {type(result)}")
            print(f"Result content: {result}")
        
        try:
            if 'graph_rag_retriever' in globals():
                print("Cleaning up graph_rag_retriever...")
                graph_rag_retriever.cleanup()
                print("Cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")