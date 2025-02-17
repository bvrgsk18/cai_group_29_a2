import streamlit as st
import sys
from pathlib import Path
from src.config import *
from src.data_processor import FinancialDataProcessor
from src.embeddings import EmbeddingManager
from src.retriever import HybridRetriever
from src.guardrails import Guardrails

# Streamlit page config
st.set_page_config(
    page_title="Financial Statement Q&A",
    page_icon="💼",
    layout="wide"
)

def initialize_system():
    """Initialize the RAG system components."""
    if 'processor' not in st.session_state:
        st.session_state.processor = FinancialDataProcessor()
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager(SBERT_MODEL, CACHE_DIR)
    if 'retriever' not in st.session_state:
        st.session_state.retriever = HybridRetriever(st.session_state.embedding_manager, CACHE_DIR)
    if 'guardrails' not in st.session_state:
        st.session_state.guardrails = Guardrails()

def process_document(file):
    """Process uploaded document and build indices."""
    with st.spinner("Processing document..."):
        # Extract text from PDF
        text = st.session_state.processor.read_pdf(file)
        chunks = st.session_state.processor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Build indices
        embeddings = st.session_state.embedding_manager.embed_texts(chunks)
        st.session_state.embedding_manager.build_index(embeddings)
        st.session_state.retriever.init_bm25(chunks)
        
        # Save indices for future use
        st.session_state.embedding_manager.save_index('financial_index.faiss')
        st.session_state.retriever.save_state('retriever_state.pkl')
        
        return len(chunks)

def main():
    st.title("📊 Financial Statement Question Answering System")
    st.write("Upload a financial statement PDF and ask questions about it.")
    
    # Initialize system
    initialize_system()
    
    # File upload section
    with st.container():
        uploaded_file = st.file_uploader("Upload Financial Statement (PDF)", type="pdf")
        
        if uploaded_file:
            if 'processed' not in st.session_state:
                num_chunks = process_document(uploaded_file)
                st.session_state.processed = True
                st.success(f"Document processed into {num_chunks} chunks!")
    
    # Query section
    if 'processed' in st.session_state:
        st.divider()
        query = st.text_input("💭 Ask a question about the financial statement:")
        
        if query:
            # Input validation
            is_valid, message = st.session_state.guardrails.validate_input(query)
            
            if is_valid:
                with st.spinner("Searching for answer..."):
                    # Retrieve relevant chunks
                    results = st.session_state.retriever.hybrid_search(query, TOP_K)
                    
                    # Display results
                    st.subheader("📝 Answer")
                    for i, (chunk, score) in enumerate(results, 1):
                        with st.expander(f"Source {i} (Relevance: {score:.2f})"):
                            st.write(chunk)
            else:
                st.error(message)
    
    # Instructions
    if 'processed' not in st.session_state:
        st.info("👆 Start by uploading a financial statement PDF.")

if __name__ == "__main__":
    main()