import streamlit as st
import os
from pypdf import PdfReader
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="Chandrayaan RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components"""
    # Initialize ChromaDB
    client = Client(Settings(persist_directory='./chroma_db'))
    
    # Get or create collection
    collection_name = "chandrayaan_mission_rag"
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)
    
    # Load embedder
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load tokenizer and model
    tokeni = AutoTokenizer.from_pretrained('google/flan-t5-base')
    mdl = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    
    # Create text generation pipeline
    generate = pipeline('text2text-generation', model=mdl, tokenizer=tokeni)
    
    return client, collection, embedder, generate

@st.cache_resource
def get_tokenizer():
    """Get tokenizer for chunking"""
    return AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def clean_text(text):
    """Clean extracted text from PDFs"""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_chunks(text, tokenizer, max_tokens=150, overlap=40):
    """Create overlapping chunks from text"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chktok = tokens[start:end]
        chltxt = tokenizer.decode(chktok, skip_special_tokens=True)
        chunks.append(chltxt)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def rag_query(query, embedder, collection, generate, top_k=3, max_length=289):
    """Query the RAG system"""
    q_emb = embedder.encode(query).tolist()
    top = collection.query(query_embeddings=[q_emb], n_results=top_k)
    top_chks = top['documents'][0]
    
    context = '\n\n'.join(top_chks)
    
    pmt = f"""You are an expert on the Chandrayaan-1 and Chandrayaan-2 missions and their scientific payloads.
Use the following retrieved context to answer the question as accurately as possible.
If the answer is not found in the context, state that the information is not available in the provided document.
Context:
{context}
Question: {query}
Answer in a complete sentence:"""
    
    ans = generate(pmt, max_length=max_length, min_length=5, do_sample=False)[0]['generated_text']
    return ans, top_chks

# Header
st.markdown("""
    <div class="header-container">
    <h1>🚀 Chandrayaan Mission RAG System</h1>
    <p>Ask questions about India's Chandrayaan lunar missions and get AI-powered answers based on official documentation</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Number of retrieved documents", min_value=1, max_value=10, value=3, 
                      help="How many relevant documents to retrieve for context")
    max_length = st.slider("Maximum answer length", min_value=50, max_value=500, value=289,
                          help="Maximum length of generated answer")

# Initialize RAG system
with st.spinner("Loading RAG system..."):
    try:
        client, collection, embedder, generate = initialize_rag_system()
        st.sidebar.success("✅ RAG System Ready")
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

# Main query interface
st.subheader("📝 Ask a Question")

# Query input
query = st.text_area(
    "Enter your question about the Chandrayaan missions:",
    placeholder="e.g., What was the scientific objective of the Terrain Mapping Camera (TMC)?",
    height=100
)

# Query button
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    submit_button = st.button("🔍 Search", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Process query
if submit_button and query:
    with st.spinner("🔄 Processing your question..."):
        try:
            answer, sources = rag_query(query, embedder, collection, generate, top_k, max_length)
            
            # Display answer
            st.success("✅ Answer Generated")
            st.markdown("### 📖 Answer")
            st.write(answer)
            
            # Display sources
            with st.expander("📚 Retrieved Context (Source Documents)"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Document {i}:**")
                    st.info(source)
        
        except Exception as e:
            st.error(f"Error processing query: {e}")
elif submit_button:
    st.warning("Please enter a question first.")

# Example queries
st.markdown("---")
st.subheader("💡 Example Questions")
examples = [
    "What was the scientific objective of the Terrain Mapping Camera (TMC)?",
    "What two minor elements did the CLASS instrument detect remotely for the first time in the near-side Mare?",
    "What are the main goals of the Chandrayaan-2 mission?",
    "Tell me about the payloads onboard Chandrayaan-1."
]

for example in examples:
    if st.button(f"❓ {example}", use_container_width=True, key=example):
        with st.spinner("🔄 Processing..."):
            try:
                answer, sources = rag_query(example, embedder, collection, generate, top_k, max_length)
                st.success("✅ Answer Generated")
                st.markdown("### 📖 Answer")
                st.write(answer)
                with st.expander("📚 Retrieved Context"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Document {i}:**")
                        st.info(source)
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
    <p>🛰️ Chandrayaan RAG System | Powered by Streamlit, ChromaDB, and Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)
