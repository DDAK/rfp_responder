from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.tools import Tool
from pathlib import Path
import io
import os
import fitz
import uuid
import tempfile
from minio import Minio
import psycopg2 as psycopg
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1. Document Parsing & Indexing ==============================================

# Initialize MinIO client
minio_client = Minio(
    endpoint=os.environ.get("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minio"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minio123"),
    secure=False  # Set to True if using HTTPS
)

bucket_name = "data-bucket"
context_docs = []
file_names = [
    "Baseline_Model-_Contract__Large_Business__Dec_2024.pdf", 
    "Baseline_Model-_Contract__Small_Business__Dec_2024.pdf", 
    "HR001125S0009-Amendment-01.pdf", 
    "HR001125S0009-Amendment-02.pdf",
    "HR001125S0009.pdf",
]


# ─── Helper: download file from MinIO to temp file ---------------------------
def list_minio_files(prefix=""):
    objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]

def download_from_minio(object_name):
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    try:
        # Get object data
        response = minio_client.get_object(bucket_name, object_name)
        # Write to temp file
        for d in response.stream(32*1024):
            temp_file.write(d)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise Exception(f"Error downloading {object_name}: {str(e)}")


# Create a temporary directory to store downloaded files
temp_dir = "./temp_files"
os.makedirs(temp_dir, exist_ok=True)

for file_name in list_minio_files():
    if file_name not in file_names:
        continue
    
    temp_file_path = download_from_minio(file_name)

    # Load the document
    loader = PyMuPDF4LLMLoader(
        temp_file_path,
        mode="single",
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_name  # Store just the filename as source
    context_docs.extend(docs)

# Create pgvector store with HuggingFace embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(context_docs)

# Using open-source embeddings (Groq doesn't provide native embeddings)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# PostgreSQL connection string
CONNECTION_STRING = os.environ.get(
    "PG_CONNECTION_STRING", 
    "postgresql+psycopg://demo:demo1234@postgres:5432/rag"
)

# Custom implementation to store documents in the custom schema
class CustomPGStore:
    def __init__(self, connection_string, embedding_model):
        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.conn = self._connect()
        
    def _connect(self):
        conn_parts = self.connection_string.split("://")[1].split("@")
        user_pass = conn_parts[0].split(":")
        host_db = conn_parts[1].split("/")
        host_port = host_db[0].split(":")
        
        # Extract connection parameters
        user = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ""
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else "5432"
        dbname = host_db[1]
        
        return psycopg.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
    
    def add_documents(self, documents):
        """Add documents to the custom PG table with embeddings"""
        with self.conn.cursor() as cursor:
            # Generate embeddings for all texts
            texts = [doc.page_content for doc in documents]
            embeddings_list = self.embedding_model.embed_documents(texts)
            
            # Insert each document
            for i, doc in enumerate(documents):
                chunk_id = str(uuid.uuid4())
                pdf_name = doc.metadata.get("source", "")
                text = doc.page_content
                embedding = embeddings_list[i]
                doc_type = "rfp"  # Default document type
                case_id = ""  # Default case ID
                extra = {}  # Additional metadata as JSON
                
                # Add the rest of metadata to extra
                for key, value in doc.metadata.items():
                    if key != "source":
                        extra[key] = value
                
                cursor.execute(
                    """
                    INSERT INTO documents 
                    (chunk_id, pdf_name, text, embedding, doc_type, case_id, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (chunk_id, pdf_name, text, embedding, doc_type, case_id, extra)
                )
            
            self.conn.commit()
            
    def similarity_search(self, query, k=5, filter=None):
        """Search for documents similar to the query"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        with self.conn.cursor() as cursor:
            sql_query = """
                SELECT chunk_id, pdf_name, text, extra 
                FROM documents
                WHERE 1=1
            """
            params = []
            
            # Add filter condition if specified
            if filter and "source" in filter:
                sql_query += " AND pdf_name = %s"
                params.append(filter["source"])
            
            # Add similarity search
            sql_query += """
                ORDER BY embedding <=> %s
                LIMIT %s
            """
            params.extend([query_embedding, k])
            
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            # Convert results to Document objects
            documents = []
            for chunk_id, pdf_name, text, extra in results:
                metadata = {"source": pdf_name}
                if extra:
                    metadata.update(extra)
                documents.append(Document(page_content=text, metadata=metadata))
                
            return documents
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever object for the store"""
        search_kwargs = search_kwargs or {}
        
        def _retrieve(query):
            return self.similarity_search(query, **search_kwargs)
        
        return _retrieve

# Initialize custom store and add documents
vector_store = CustomPGStore(CONNECTION_STRING, embeddings)
vector_store.add_documents(split_docs)

# Create retriever functions
def source_retriever(source: str):
    """Create retriever filtered by source"""
    def _retrieve(query: str):
        return vector_store.similarity_search(
            query, 
            k=5,
            filter={"source": source}
        )
    return _retrieve

tools = [
    Tool.from_function(
        func=source_retriever(file_name),
        name=f"{Path(file_name).stem}_retriever",
        description=f"Retrieves content from {file_name} file"
    ) for file_name in file_names
]