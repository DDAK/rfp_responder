# app.py  – simple Streamlit UI for rag_service.py
import os, requests, uuid, streamlit as st
from PIL import Image

SESSION_ID = st.session_state.get("sid") or str(uuid.uuid4())
st.session_state["sid"] = SESSION_ID

API_URL = os.getenv("RAG_API_URL", "http://rag-api:8000")  # container name
CHUNK_URL = f"{API_URL}/chunk"        # endpoint added below

# ---------- helper ---------------------------------------------------------
def ask_rag(question, session_id, top_k=6):
    r = requests.post(f"{API_URL}/ask",
        json={"question": question, "session_id": session_id, "k": top_k},
        timeout=90)
    r.raise_for_status()
    return r.json()

def fetch_chunk(chunk_id: str):
    r = requests.get(f"{CHUNK_URL}/{chunk_id}", timeout=30)
    return r.json().get("text", "")

# ---------- page -----------------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="Knowledge Graph-Enhanced RAG System",
    page_icon="🧠",
    layout="wide"
)


# Custom function for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Try to load local CSS if available
try:
    local_css("style.css")
except:
    # Fallback inline CSS
    st.markdown("""
    <style>
    .header-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .process-step {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .key-component {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section
st.title("🤖 Social‑Support RAG Assistant")
st.subheader("A hybrid approach combining vector search with knowledge graph filtering")


# Function to display local images or fallback to placeholder
def display_image(image_path, caption=""):
    try:
        image = Image.open(image_path)
        resized_image = image.resize((600, 800))
        st.image(resized_image, caption=caption)
    except Exception as e:
        raise e
        st.warning(f"Could not load image from {image_path}. Using diagram instead.")
        # Display diagram code


# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("## System Overview")
    st.write("""
    This system implements a hybrid Retrieval-Augmented Generation (RAG) approach that combines vector 
    similarity search with knowledge graph filtering to provide accurate answers to queries about 
    social security documents and applications.
    """)

    st.markdown("## Assumptions")
    st.write("""
        1. **Document Structure**: Documents follow predictable formats that can be classified

        2. **Query Patterns**: Users often refer to specific entities (people, case IDs, organizations)

        3. **Performance Requirements**: Token budget management is necessary for optimal LLM performance

        4. **Data Relationships**: Financial information, employment status, and case relationships are key for answering queries
        """)

    st.markdown("## Key Components")

    with st.expander("Document Processing Pipeline", expanded=False):
        st.markdown('<div class="key-component">', unsafe_allow_html=True)
        st.write("""
        - PDF documents are classified into different types (policy, worksheet, affidavit, salary slip)
        - Documents are chunked into semantic units (~150 tokens each)
        - Embeddings are generated for each chunk using Ollama/Llama models
        - Extracted chunks are stored in Postgres with their embeddings
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Knowledge Graph Construction", expanded=False):
        st.markdown('<div class="key-component">', unsafe_allow_html=True)
        st.write("""
        - Triples (subject-predicate-object relationships) are extracted from documents
        - Relationships include employment, income data, residency, etc.
        - Neo4j graph database stores entity relationships
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Hybrid Retrieval Service", expanded=False):
        st.markdown('<div class="key-component">', unsafe_allow_html=True)
        st.write("""
        - Extracts entities from queries using SpaCy
        - Filters relevant case IDs using the knowledge graph
        - Performs vector similarity search on filtered document chunks
        - Manages token budget to optimize context size
        - Summarizes chunks when necessary to fit token limitations
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Session Management", expanded=False):
        st.markdown('<div class="key-component">', unsafe_allow_html=True)
        st.write("""
        - Redis stores conversation history
        - Maintains context across multiple interactions
        """)
        st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown("## System Architecture")
    # Try to display the flow diagram image, otherwise show placeholder
    display_image("flow_diagram.png", "Data Flow Diagram")


if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history render
for msg in st.session_state.messages:
    role, text = msg
    st.chat_message(role).write(text, unsafe_allow_html=True)

# user input
if prompt := st.chat_input("Type your question…"):
    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)

    with st.spinner("Retrieving…"):
        try:
            res = ask_rag(prompt, SESSION_ID)
            answer = res["answer"]
            sources = res["sources"]
        except Exception as e:
            answer = f"❌ error: {e}"
            sources = []

    # show assistant answer
    st.session_state.messages.append(("assistant", answer))
    st.chat_message("assistant").write(answer)

    # expandable context
    if sources:
        with st.expander("Source snippets", expanded=False):
            for sid in sources:
                chunk = fetch_chunk(sid)
                st.markdown(f"**{sid[:8]}…**")
                st.code(chunk, language="markdown")