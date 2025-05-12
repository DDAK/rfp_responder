import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, PrivateAttr

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger
from lightrag.kg.shared_storage import initialize_pipeline_status

from langchain.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document

# Ensure storage directory exists
os.makedirs("./rag_storage", exist_ok=True)

# Environment variables
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


class LightRAGRetriever(BaseRetriever, BaseModel):
    """
    A synchronous LangChain Retriever using LightRAG.
    """
    mode: str = "hybrid"
    _rag: LightRAG = PrivateAttr()
    _loop: Optional[asyncio.AbstractEventLoop] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Create and set event loop
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # initialize LightRAG
        self._rag = LightRAG(
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
                    host="http://127.0.0.1:11434",
                ),
            ),
            kv_storage="MongoKVStorage",
            graph_storage="Neo4JStorage",
            vector_storage="PGVectorStorage",
            doc_status_storage="PGDocStatusStorage",
        )
        # initialize storages synchronously
        self._loop.run_until_complete(self._rag.initialize_storages())
        # initialize pipeline status synchronously
        self._loop.run_until_complete(initialize_pipeline_status())

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from LightRAG synchronously."""
        try:
            param = QueryParam(mode=self.mode)
            prompt = f"Query:{query}"
            # run async query synchronously using the instance's event loop
            response = self._loop.run_until_complete(self._rag.aquery(prompt, param=param))
            return [Document(page_content=response, metadata={})]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def cleanup(self):
        """Cleanup resources synchronously."""
        try:
            if self._loop and self._loop.is_running():
                self._loop.run_until_complete(self._rag.finalize_storages())
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            if self._loop:
                self._loop.close()

    def __del__(self):
        self.cleanup()


# Instantiate retriever and tool
graph_rag_retriever = LightRAGRetriever(mode="hybrid")

graphrag_tool = create_retriever_tool(
    retriever=graph_rag_retriever,
    name="graph_rag_qa",
    description="Use this tool to query our RFP knowledge graph.",
)

class TestGraphRAGRetriever(BaseRetriever, BaseModel):
    """
    A synchronous LangChain Retriever using LightRAG.
    """
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return [
            Document(
                page_content="The Securing Artificial Intelligence for Battlefield Effective Robustness \
                         (SABER) solicitation (Notice ID HR001125S0009) is an active DARPA opportunity published \
                         March 12, 2025 and last updated April 23, 2025, with final proposals due June 3, 2025. \
                         Issued by the Department of Defense's Defense Advanced Research Projects Agency, this \
                         Applied Research contract (NAICS 541715; PSC AC12) seeks to advance the security and \
                         reliability of AI-enabled systems under realistic battlefield conditions.",
                metadata={}
            ),
            Document(
                page_content="Modern machine-learning models can offer significant decision-making \
                         advantages, but they remain vulnerable to distributional shifts and adversarial \
                         manipulation—ranging from data-poisoning attacks and physically constrained adversarial \
                         patches to model-stealing exploits. While much of the research to date has been conducted \
                         in highly controlled environments, DARPA notes a critical gap: there is currently no \
                         practical, operational capability to red-team deployed military AI systems and uncover \
                         their adversarial weaknesses in real-world settings.",
                metadata={}
            ),
            Document(
                page_content="SABER's goal is to stand up a robust, sustainable AI red-teaming capability \
                         within the DoD that can systematically assess autonomous ground and aerial systems slated \
                         for deployment over the next 1–3 years. The program will fund performers who can survey and \
                         evaluate state-of-the-art counter-AI techniques, develop novel physical and cyber adversarial \
                         tools, and integrate these into an 'AI red team' workflow capable of end-to-end operational testing.",
                metadata={}
            ),
            Document(
                page_content="Successful awardees will help define and execute a repeatable process for AI red \
                         teaming: selecting target platforms, crafting and deploying adversarial attacks under \
                         battlefield-relevant constraints, measuring system degradations, and recommending hardening \
                         strategies. In doing so, SABER aims not only to reveal the currently unknown security risks of \
                         fielded AI systems but also to seed a long-term, institutionalized red-teaming model for the \
                         Department of Defense.",
                metadata={}
            )
        ]

test_graph_rag_retriever = TestGraphRAGRetriever()
test_graphrag_tool = create_retriever_tool(
    retriever=test_graph_rag_retriever,
    name="test_graph_rag_qa",
    description="Use this tool to query our RFP knowledge graph.",
)
