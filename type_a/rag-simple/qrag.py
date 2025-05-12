import os
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger
from langchain.tools import Tool

async def query_knowledge_graph(q: str, storage_dir: str):
    """
    """
    # Ensure storage directory exists
    os.makedirs(storage_dir, exist_ok=True)

    # Initialize LightRAG with Neo4jStorage
    BATCH_SIZE_NODES = 500
    BATCH_SIZE_EDGES = 100
    os.environ.setdefault("NEO4J_URI", os.getenv("NEO4J_URI", "neo4j://localhost:7687"))
    os.environ.setdefault("NEO4J_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j"))
    os.environ.setdefault("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "mypassword"))
    
    # mongo
    os.environ.setdefault("MONGO_URI", os.getenv("MONGO_URI", "mongodb://mongoUser:mongoPass@localhost:27017/"))
    os.environ.setdefault("MONGO_DATABASE", os.getenv("MONGO_DATABASE", "LightRAG"))

    # pgvector
    os.environ.setdefault("POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost"))
    os.environ.setdefault("POSTGRES_PORT", os.getenv("POSTGRES_PORT", "15432"))
    os.environ.setdefault("POSTGRES_USER", os.getenv("POSTGRES_USER", "demo"))
    os.environ.setdefault("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "demo1234"))
    os.environ.setdefault("POSTGRES_DATABASE", os.getenv("POSTGRES_DATABASE", "rag"))

    rag = LightRAG(
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
        # auto_manage_storages_states=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    param = QueryParam(mode="hybrid",
                        # only_need_context=True, # no LLM call
                        # only_need_prompt=True,
                        top_k=10,               # how many graph hits
                        # you can tune these token limits as needed:
                        # max_token_for_text_unit=1200,
                        # max_token_for_global_context=2000,
                        )
 # Test different query modes
    queries = [
        # ("Naive",  QueryParam(mode="naive"),  "What are the requirements for the proposal (rfp or rfi)? And list those requirements in a table format."),
        # ("Local",  QueryParam(mode="local"),  "What are the requirements for the proposal (rfp or rfi)? And list those requirements in a table format."),
        # ("Global", QueryParam(mode="global"), "What are the requirements for the proposal (rfp or rfi)? And list those requirements in a table format."),
        ("Hybrid", param, q),
    ]
    for label, param, q in queries:
        print(f"\n{label} Search:")
        # call the async API directly
        answer = await rag.aquery(q, param=param)
        print(answer)
    await rag.finalize_storages()


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Ingest documents into a LightRAG knowledge graph stored in Neo4j."
    # )
    # parser.add_argument(
    #     "--data-dir", required=True,
    #     help="Directory containing files to ingest (pdf, docx, xlsx, md, txt)."
    # )
    # parser.add_argument(
    #     "--storage-dir", required=True,
    #     help="Directory for LightRAG working/storage files."
    # )
    # args = parser.parse_args()
    # # print(args.data_dir, args.storage_dir)
    rag = asyncio.run(query_knowledge_graph("What are the requirements for the SABER rfp?", "./rag_storage"))
  