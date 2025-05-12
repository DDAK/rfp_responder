from langchain.document_loaders import PyPDFLoader, DocxLoader, CSVLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from crewai import Agent, Task, Crew, Process
from langchain.llms import ChatOpenAI
import zipfile
import os
import json

class DocumentRouter:
    """Routes documents to appropriate processors based on file type"""
    
    def __init__(self):
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': DocxLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }
    
    def route_document(self, file_path):
        """Route document to appropriate loader"""
        _, ext = os.path.splitext(file_path)
        
        if ext == '.zip':
            # Extract zip and route each file
            extract_dir = self._extract_zip(file_path)
            return self._process_directory(extract_dir)
        
        if ext.lower() in self.loaders:
            return self.loaders[ext.lower()](file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
    def _extract_zip(self, zip_path):
        """Extract zip file to temporary directory"""
        extract_dir = f"temp_extract_{os.path.basename(zip_path)}"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    
    def _process_directory(self, directory):
        """Process all files in a directory"""
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    loader = self.route_document(file_path)
                    documents.extend(loader.load())
                except ValueError:
                    print(f"Skipping unsupported file: {file_path}")
        return documents


class RFPIngestionPipeline:
    """Main ingestion pipeline for RFP documents"""
    
    def __init__(self, db_connection_string, neo4j_uri, neo4j_username, neo4j_password):
        self.document_router = DocumentRouter()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.db_conn = db_connection_string
        
        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Initialize language model
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Initialize agents
        self._setup_agents()
        
    def _setup_agents(self):
        """Setup the agent crew for RFP processing"""
        
        # Document extraction agent
        extraction_agent = Agent(
            role="Document Extraction Specialist",
            goal="Extract all text and metadata from documents accurately",
            backstory="I am an expert at parsing complex documents in various formats",
            verbose=True,
            llm=self.llm,
            tools=[]  # We'll use built-in tools
        )
        
        # Requirements identification agent
        requirements_agent = Agent(
            role="Requirements Specialist",
            goal="Identify all requirements in RFP documents",
            backstory="I excel at finding 'shall', 'must', 'should' statements and categorizing them",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        # Evaluation criteria agent
        evaluation_agent = Agent(
            role="Evaluation Criteria Analyst",
            goal="Extract all evaluation criteria and their weights",
            backstory="I specialize in understanding how proposals will be judged",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        # Submission specialist agent
        submission_agent = Agent(
            role="Submission Specialist",
            goal="Extract all submission instructions and deadlines",
            backstory="I am meticulous about understanding formatting and submission requirements",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        # Extract documents task
        extract_task = Task(
            description="Extract all text and metadata from the provided documents",
            agent=extraction_agent,
            expected_output="A dictionary containing all extracted text and metadata"
        )
        
        # Find requirements task
        requirements_task = Task(
            description="Identify all requirements in the RFP documents. Look for 'shall', 'must', 'should', etc.",
            agent=requirements_agent,
            expected_output="A JSON array of all requirements, with references to their location in the document"
        )
        
        # Extract evaluation criteria task
        evaluation_task = Task(
            description="Extract all evaluation criteria and their weightings from the RFP",
            agent=evaluation_agent,
            expected_output="A JSON object containing evaluation criteria, weightings, and methodology"
        )
        
        # Extract submission instructions task
        submission_task = Task(
            description="Extract all submission instructions, deadlines, and formatting requirements",
            agent=submission_agent,
            expected_output="A JSON object with deadlines, formatting requirements, and submission methods"
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[extraction_agent, requirements_agent, evaluation_agent, submission_agent],
            tasks=[extract_task, requirements_task, evaluation_task, submission_task],
            verbose=2,
            process=Process.sequential
        )
    
    def process_document(self, file_path):
        """Process a document through the ingestion pipeline"""
        
        # Route document to appropriate loader
        documents = self.document_router.route_document(file_path)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Store in vector database
        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            connection_string=self.db_conn,
            collection_name=f"rfp_{os.path.basename(file_path)}"
        )
        
        # Process with agent crew
        results = self.crew.kickoff(inputs={"documents": documents})
        
        # Create knowledge graph from extracted information
        self._build_knowledge_graph(results)
        
        return {
            "vector_store_id": f"rfp_{os.path.basename(file_path)}",
            "extraction_results": results
        }
    
    def _build_knowledge_graph(self, extraction_results):
        """Build knowledge graph from extracted information"""
        
        # Extract components from results
        requirements = json.loads(extraction_results["requirements_task"])
        evaluation_criteria = json.loads(extraction_results["evaluation_task"])
        submission_info = json.loads(extraction_results["submission_task"])
        
        # Create RFP node
        self.graph.query("""
        CREATE (rfp:RFP {
            id: $rfp_id,
            title: $title,
            due_date: $due_date
        })
        """, params={
            "rfp_id": extraction_results["metadata"]["solicitation_number"],
            "title": extraction_results["metadata"]["title"],
            "due_date": submission_info["deadline"]
        })
        
        # Create requirement nodes and relationships
        for req in requirements:
            self.graph.query("""
            MATCH (rfp:RFP {id: $rfp_id})
            CREATE (req:Requirement {
                id: $req_id,
                text: $text,
                type: $type,
                location: $location
            })
            CREATE (rfp)-[:HAS_REQUIREMENT]->(req)
            """, params={
                "rfp_id": extraction_results["metadata"]["solicitation_number"],
                "req_id": req["id"],
                "text": req["text"],
                "type": req["type"],  # must have, should have, etc
                "location": req["location"]
            })
        
        # Create evaluation criteria nodes and relationships
        for criterion in evaluation_criteria["criteria"]:
            self.graph.query("""
            MATCH (rfp:RFP {id: $rfp_id})
            CREATE (ec:EvaluationCriterion {
                name: $name,
                weight: $weight,
                description: $description
            })
            CREATE (rfp)-[:EVALUATED_BY]->(ec)
            """, params={
                "rfp_id": extraction_results["metadata"]["solicitation_number"],
                "name": criterion["name"],
                "weight": criterion["weight"],
                "description": criterion["description"]
            })
            
        # Create submission instructions node
        self.graph.query("""
        MATCH (rfp:RFP {id: $rfp_id})
        CREATE (si:SubmissionInstructions {
            deadline: $deadline,
            method: $method,
            format: $format,
            page_limit: $page_limit
        })
        CREATE (rfp)-[:HAS_SUBMISSION_INSTRUCTIONS]->(si)
        """, params={
            "rfp_id": extraction_results["metadata"]["solicitation_number"],
            "deadline": submission_info["deadline"],
            "method": submission_info["method"],
            "format": submission_info["format"],
            "page_limit": submission_info["page_limit"]
        })


class ProposalStyleIngestionPipeline:
    """Ingestion pipeline for prior proposals to learn style and structure"""
    
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, db_connection_string):
        self.document_router = DocumentRouter()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.db_conn = db_connection_string
        
        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Initialize language model
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
    def process_proposal(self, file_path, metadata):
        """Process a prior proposal document"""
        
        # Route document to appropriate loader
        documents = self.document_router.route_document(file_path)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Store in vector database
        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            connection_string=self.db_conn,
            collection_name=f"prop_{os.path.basename(file_path)}"
        )
        
        # Extract style and structure information
        style_info = self._extract_style_info(documents, metadata)
        
        # Build knowledge graph for this proposal
        self._build_proposal_knowledge_graph(style_info, metadata)
        
        return {
            "vector_store_id": f"prop_{os.path.basename(file_path)}",
            "style_info": style_info
        }
    
    def _extract_style_info(self, documents, metadata):
        """Extract style and structure information from proposal"""
        
        # Using LLM to analyze style
        style_prompt = f"""
        Analyze the following proposal document for style and structure.
        Extract the following information:
        1. Overall tone (formal, technical, persuasive, etc.)
        2. Section structure and organization
        3. Common phrases and terminology
        4. Formatting patterns (bullets, numbering, etc.)
        5. Graphics and visual elements usage
        
        Metadata about this proposal:
        - Title: {metadata['title']}
        - Client: {metadata['client']}
        - Result: {metadata['result']} (won/lost/pending)
        
        Return your analysis as a structured JSON object.
        """
        
        combined_text = "\n\n".join([doc.page_content for doc in documents[:5]])  # First 5 chunks for analysis
        
        result = self.llm.invoke(style_prompt + combined_text)
        
        # Parse the JSON response
        try:
            style_info = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            style_info = {
                "tone": "unknown",
                "structure": "unknown",
                "phrases": [],
                "formatting": "unknown",
                "visuals": "unknown"
            }
            
        return style_info
    
    def _build_proposal_knowledge_graph(self, style_info, metadata):
        """Build knowledge graph for proposal style information"""
        
        # Create proposal node
        self.graph.query("""
        CREATE (p:Proposal {
            id: $id,
            title: $title,
            client: $client,
            result: $result,
            date: $date
        })
        """, params={
            "id": metadata["id"],
            "title": metadata["title"],
            "client": metadata["client"],
            "result": metadata["result"],
            "date": metadata["date"]
        })
        
        # Create style node
        self.graph.query("""
        MATCH (p:Proposal {id: $id})
        CREATE (s:Style {
            tone: $tone,
            structure: $structure,
            formatting: $formatting,
            visuals: $visuals
        })
        CREATE (p)-[:HAS_STYLE]->(s)
        """, params={
            "id": metadata["id"],
            "tone": style_info["tone"],
            "structure": style_info["structure"],
            "formatting": style_info["formatting"],
            "visuals": style_info["visuals"]
        })
        
        # Create phrase nodes
        for phrase in style_info.get("phrases", []):
            self.graph.query("""
            MATCH (p:Proposal {id: $id})
            CREATE (ph:Phrase {text: $text})
            CREATE (p)-[:USES_PHRASE]->(ph)
            """, params={
                "id": metadata["id"],
                "text": phrase
            })
            
        # Link to client
        self.graph.query("""
        MATCH (p:Proposal {id: $id})
        MERGE (c:Client {name: $client})
        CREATE (p)-[:SUBMITTED_TO]->(c)
        """, params={
            "id": metadata["id"],
            "client": metadata["client"]
        })