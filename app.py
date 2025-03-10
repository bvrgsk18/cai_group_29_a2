import streamlit as st
import asyncio
from src.config import *
from src.data_processor import FinancialDataProcessor
from src.embeddings import EmbeddingManager
from src.guardrails import Guardrails
from src.llm_handler import LlmHandler
from retriever import HybridRetriever
import ollama
import re


st.set_page_config(
    page_title="CAI Group 29 - Assignment 2 - RAG Chatbot - Financial Results Q&A",
    layout="wide"
)

# Custom CSS for chat styling
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 700px;
        margin: auto;
    }
    .bot-message {
        background-color: #BAD8B6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        width: fit-content;
        max-width: 80%;
    }
    .user-message {
        background-color: #FFA725;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_system():
    """Initialize system components."""
    # Fix for asyncio event loop conflict
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # If no active loop, create one
        asyncio.set_event_loop(asyncio.new_event_loop())

    if "processor" not in st.session_state:
        st.session_state.processor = FinancialDataProcessor()
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager(SBERT_MODEL, CACHE_DIR)
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = Guardrails()
    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever(st.session_state.embedding_manager, SEARCH_DIR)
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Stores full chat history
    
    # Ensure model loads before chat starts
    if 'llm_handler' not in st.session_state or st.session_state.llm_handler is None:
        with st.spinner("Loading language model... This may take a few minutes. Please wait."):
            st.session_state.llm_handler = LlmHandler(MAX_NEW_TOKENS)

    # Display confirmation message once the model is ready
    st.session_state.model_ready = True
    st.success("Language model loaded successfully!")

def process_document(file):
    """Process uploaded document and build vector index."""
    with st.spinner("Processing document..."):
        text = st.session_state.processor.read_pdf(file)
        cleaned_text = re.sub(r"\s+", " ", text).strip().lower()
        chunks = st.session_state.processor.chunk_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        embeddings = st.session_state.embedding_manager.embed_texts(chunks)

        if embeddings is None or len(embeddings) == 0:
            st.error(" Embeddings could not be generated. Please check your input text.")
            return

        st.session_state.embedding_manager.build_index(embeddings)
        st.session_state.embedding_manager.save_index("financial_index.faiss")

        st.session_state.retriever.init_bm25(chunks)
        st.session_state.retriever.save_state("bm25_state_temp.txt")

        st.session_state.chunks = chunks
        st.session_state.processed = True
        st.success(f"Document processed into {len(chunks)} chunks!")

def classify_confidence(score):
    """Classify confidence score into High, Medium, or Low categories."""
    if score >= 0.75:
        return "🔵 High Confidence"
    elif score >= 0.5:
        return "🟠 Medium Confidence"
    else:
        return "🔴 Low Confidence"

def main():

    # Add background color to the main page
    st.markdown(
        """
        <style>
        .main {
            background-color: #F8F3D9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center; color: #881148;'>CAI Group 29 - Assignment 2 - RAG ChatBot</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;color: #D98324;'>Financial Statement Question Answering System</h1>", unsafe_allow_html=True)

    # CAI Group 29 Info
    st.sidebar.title("CAI Group 29 Members")
    table_md = """
    | Name               | ID          |
    |--------------------|------------|
    | B V R G S Kumar   | 2023AA05013 |
    | ADARSH VADAPALLI  | 2023AA05090 |
    | DINESH KUMAR      | 2023AA05096 |
    | CYNTHIA R         | 2023AA05100 |
    | TARUSH JAISWAL    | 2023AA05769 |
    """
    st.sidebar.markdown(table_md)

    st.sidebar.title("Implementation Details")
    st.sidebar.markdown(f"""
    | Task               | Details    |
    |--------------------|------------|
    | Embedding Model   | BAAI ({SBERT_MODEL}) |
    | LLM for Response Generation  | TinyLlama (TinyLlama-1.1B-Chat-v1.0a) |
    | Chunk Size      | {CHUNK_SIZE} tokens |
    | Sparse Retrieval | BM25 |
    | Dense Retrieval  | FAISS |
    | Retrieval        | Hybrid Search |
    """)

    # Add background color to sidebar and set font color to white
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #881148;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] table {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Initialize system
    initialize_system()
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Financial Statement (PDF)", type="pdf")
    if uploaded_file and "processed" not in st.session_state:
        process_document(uploaded_file)

    # Chat interface
    if "processed" in st.session_state:
        st.divider()

        # Display Full Chat History with Background Colors
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.conversation:
            role = message["role"]
            content = message["content"]
            css_class = "user-message" if role == "user" else "bot-message"
            st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # User query input
        query = st.chat_input("Ask a question about the financial statement...")


        if query:
            # Append User's Question to Chat History
            st.session_state.conversation.append({"role": "user", "content": f"{query}"})
            # Display a placeholder response immediately
            placeholder = st.empty()
            placeholder.markdown('<div class="bot-message">Processing your request...</div>', unsafe_allow_html=True)
            
            # Process Query and Append Answer Before UI Refresh
            is_valid, message = st.session_state.guardrails.validate_input(query)
            answer, confidence_label = "", "🔴 Low Confidence"

            if not is_valid:
                answer = message
            else:
                results = st.session_state.retriever.hybrid_search(query, TOP_K)
                if results:
                    context, score = results[0]  # Score from retrieval
                    confidence_label = classify_confidence(score)

                    prompt = f"""You are a financial analyst. Answer the question from given context only. strictly no hallucinated answers.
                    Do not approximate values. Give extact numbers from context. 
                    If you do not know the answer, respond with "I'm sorry, but I couldn't find relevant information in the document.". 
                    Use the following information:
                    \n
                    Context: {context}
                    \n
                    Question: {query}
                    \n
                    Answer: """

                    # Generate Response Using LlmHandler
                    response = st.session_state.llm_handler.generate_response(prompt)
                    answer = response
                else:
                    answer = "I'm sorry, but I couldn't find relevant information in the document."

            # Append Assistant's Response Before UI Refresh
            if answer:
                final_response = f"{answer} ({confidence_label})"
            else:
                final_response = "I'm sorry, but I couldn't find relevant information in the document."
            st.session_state.conversation.append({"role": "assistant", "content": final_response})

            # Ensure UI updates correctly only AFTER storing messages
            st.rerun()

    # Reset button (clears chat but keeps document)
    if "processed" in st.session_state and st.button("🔄 Reset Chat"):
        st.session_state.conversation = []
        st.rerun()

if __name__ == "__main__":
    main()
