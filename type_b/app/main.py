import os
import json

from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from prompts import AGENT_SYSTEM_PROMPT, EXTRACT_KEYS_PROMPT, GENERATE_OUTPUT_PROMPT
from indexer import tools
from langchain_core.pydantic_v1 import BaseModel, Field



#  Workflow Definition ======================================================
dev = True
# Initialize LLM with error handling
if not dev:
    print("prod using Groq...")
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )
else:
    print("dev going back to Ollama...")
    llm = ChatOllama(
        model="qwen2.5:latest", 
        temperature=0.1)
    

# Define state schema
class WorkflowState(TypedDict):
    rfp_text: str
    questions: List[str]
    answers: Dict[str, str]
    final_response: str

# Initialize agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    ("placeholder", "{agent_scratchpad}"),
    ("human", "{input}")
])
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Define nodes
def parse_rfp(state: WorkflowState):
    from langchain_pymupdf4llm import PyMuPDF4LLMLoader
    rfp_docs = PyMuPDF4LLMLoader("./data/HR001125S0009.pdf").load()
    return {"rfp_text": "\n".join([d.page_content for d in rfp_docs])}

# Define structured output format
class RFPQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of extracted questions")

def extract_questions(state: WorkflowState):
    # Set up parser with model
    parser = JsonOutputParser(pydantic_object=RFPQuestions)
    
    # Create prompt with format instructions
    prompt = ChatPromptTemplate.from_template(EXTRACT_KEYS_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Create chain with error handling
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "rfp_text": state["rfp_text"],
            "file_metadata": "\n".join([f"{t.name}: {t.description}" for t in tools])
        })
        print(result.solicitation.aiToolUsage.questions)
        return {"questions": result.solicitation.aiToolUsage.questions}
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        # Fallback to empty list to continue workflow
        return {"questions": []}


def answer_question(state: WorkflowState):
    answers = {}
    for question in state["questions"]:
        result = agent_executor.invoke({"input": question})
        answers[question] = result["output"]
    return {"answers": answers}

def generate_final(state: WorkflowState):
    prompt = ChatPromptTemplate.from_template(GENERATE_OUTPUT_PROMPT)
    chain = prompt | llm
    return {"final_response": chain.invoke({
        "output_template": state["rfp_text"],
        "answers": json.dumps(state["answers"])
    }).content}

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("parse_rfp", parse_rfp)
workflow.add_node("extract_questions", extract_questions)
workflow.add_node("answer_questions", answer_question)
workflow.add_node("generate_output", generate_final)

workflow.set_entry_point("parse_rfp")
workflow.add_edge("parse_rfp", "extract_questions")
workflow.add_edge("extract_questions", "answer_questions")
workflow.add_edge("answer_questions", "generate_output")
workflow.add_edge("generate_output", END)

# 4. Execution ================================================================

app = workflow.compile()
if __name__ == "__main__":
    result = app.invoke({"rfp_text": "", "questions": [], "answers": {}})
    print(result["final_response"])