import torch 
torch.classes.__path__ = []
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import tempfile
import uuid
import os
import re
import chromadb
from langchain_classic.retrievers import MergerRetriever, ContextualCompressionRetriever


os.environ["ANONYMIZED_TELEMETRY"] = "False"

def clean_latex(text: str) -> str:
    """ Clean LaTeX delimiters in the text for proper rendering in Streamlit as qwen uses escaped delimiters which streamlit does not."""
    # Replace block delimiters
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    # Replace inline delimiters
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    return text



if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.temp_collection_name = f"temp_{st.session_state.session_id}"
    st.session_state.uploaded_files = []
    st.session_state.using_temp_docs = False
    st.session_state.messages = []
    st.session_state.processing_done = True
    
    
st.set_page_config(
    page_title="Physics Paper Assistant",
    page_icon="⚛️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)





@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Get embeddings model (cached across sessions)"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource
def get_llm() -> ChatOllama:
    """Get LLM instance (cached)"""
    return ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,
        num_ctx=4096,
    )


@st.cache_resource
def load_base_vectorstore() -> Chroma:
    """Load the permanent base vector store"""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="langchain"
    )

def create_temp_vectorstore(new_documents: List[Document]) -> Chroma:
    """Creates a TRULY in-memory store for ONLY the new documents."""
    # Use EphemeralClient to stay in RAM (Faster & Safer)
    client = chromadb.EphemeralClient()
    
    return Chroma.from_documents(
        documents=new_documents,
        embedding=get_embeddings(),
        client=client,
        collection_name="user_upload"
    )
    
def create_rag_chain(vectorstore: Chroma):
    """
    Create modern RAG chain using create_retrieval_chain.
    
    Args:
        vectorstore: Chroma vector store instance
        
    Returns:
        LangChain retrieval chain
    """
    llm = get_llm()

    main_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})
    if st.session_state.using_temp_docs:
            base_vs = load_base_vectorstore()
            base_retriever = base_vs.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            final_retriever = MergerRetriever(retrievers=[main_retriever, base_retriever])
    else:
        final_retriever = main_retriever
    
    compressor = FlashrankRerank(model = "ms-marco-MiniLM-L-12-v2", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=final_retriever,
        base_compressor=compressor)
   
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful physics research assistant. Use the following context from research papers to answer the question accurately and concisely.
        LATEX RULES:
        - Use $...$ for inline math (e.g., $E=mc^2$).
        - Use $$...$$ for standalone equations on their own line .
        
        - ALWAYS use LaTeX for physical constants, variables, and formulas.
        - Ensure all Greek letters and subscripts are wrapped in LaTeX.
        
        Instructions:
        - Answer based on the provided context
        - If the context doesn't contain enough information, say so clearly
        - Cite which papers or sections your answer comes from
        - Use technical language appropriate for someone with physics background
        - Be precise with equations, terminology, and concepts

        Context:
        {context}"""),
        ("human", "{input}")])
    
   
    
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    return retrieval_chain


def handle_upload():
    """Triggered via on_change in file_uploader to ensure persistence"""
    uploaded_files = st.session_state.pdf_uploader
    st.session_state.processing_done = False
    if not uploaded_files:
        return

    with st.sidebar:
        with st.spinner("Processing documents..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for f in uploaded_files:
                    path = temp_path / f.name
                    with open(path, "wb") as buffer:
                        buffer.write(f.getbuffer())
                    loader = PyPDFLoader(str(path))
                    all_docs.extend(loader.load())
            
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = text_splitter.split_documents(all_docs)
                for c in chunks: c.metadata['source_type'] = 'uploaded'
                
                
                temp_vs = create_temp_vectorstore(chunks)
                
                st.session_state.temp_chain = create_rag_chain(temp_vs)
                st.session_state.using_temp_docs = True
                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                st.session_state.processing_done = True



def format_docs(docs: List[Document]) -> str:
    """Format documents for display"""
    return "\n\n".join(doc.page_content for doc in docs)


# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration Info")
    
    st.markdown("### Model")
    st.info("🤖 Qwen 2.5 7B")
    
    st.markdown("### Embeddings")
    st.info("BGE-Small-EN-v1.5")
    
    st.markdown("---")
    st.markdown("### 📤 Upload Additional Documents")
    st.caption("Add temporary documents to your session")
    st.caption("⚠️ Not permanently saved - resets on page refresh")
    
    uploaded_files = st.file_uploader(
        "Upload Physics PDFs", 
        type=['pdf'], 
        accept_multiple_files=True, 
        key="pdf_uploader",
        on_change=handle_upload
    )
    
    if st.session_state.uploaded_files and st.session_state.processing_done:
        st.success("✅ Knowledge updated!")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("**Uploaded this session:**")
        for fname in st.session_state.uploaded_files:
            st.caption(f"📄 {fname}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", help="Remove uploaded docs, use only base knowledge"):
                st.session_state.using_temp_docs = False
                st.session_state.uploaded_files = []
                st.session_state.last_uploaded_files = None
                if 'temp_chain' in st.session_state:
                    del st.session_state.temp_chain
                st.rerun()
        
        with col2:
            total_docs = len(st.session_state.uploaded_files)
            st.metric("Docs", total_docs)
    
    st.markdown("---")
    st.markdown("### 💬 Chat Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    **Base Knowledge**: 180 physics papers and books (permanent)
    
    **Uploaded Docs**: Session-only, auto-reset on refresh
    
    **Privacy**: All processing happens locally
    """)


# Main content
st.title("⚛️ Physics Paper Assistant")

# Show active knowledge base status
if st.session_state.using_temp_docs:
    st.info(f"Using enhanced knowledge base with **{len(st.session_state.uploaded_files)}** uploaded document(s)")
else:
    st.caption("💾 Using base knowledge base (180 publications)")

# Initialize chain
try:
    if st.session_state.using_temp_docs:
        if 'temp_chain' not in st.session_state:
            st.warning("⚠️ Temporary chain not ready. Please re-upload documents.")
            st.stop()
        chain = st.session_state.temp_chain
    else:
        # Initialize base chain
        if 'base_chain' not in st.session_state:
            with st.spinner("🔄 Loading base knowledge base..."):
                base_vectorstore = load_base_vectorstore()
                st.session_state.base_chain = create_rag_chain(base_vectorstore)
        chain = st.session_state.base_chain
        
except Exception as e:
    st.error(f"❌ Failed to initialize: {e}")
    import traceback
    with st.expander("See error details"):
        st.code(traceback.format_exc())
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    # Check if from uploaded doc
                    is_uploaded = source.get('source_type') == 'uploaded'
                    icon = "📤" if is_uploaded else "📄"
                    badge = " **(uploaded)**" if is_uploaded else ""
                    
                    st.markdown(f"**{i}.** {icon} {source['file']}{badge} (page {source['page']})")
                    st.caption(source['preview'])
                    if i < len(message["sources"]):
                        st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask about your papers...", key="chat_input", disabled=st.session_state.processing_done is False ):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from the database and thinking"):
            start_time = time.time()
            response_placeholder = st.empty()
            full_answer = ""
            context_docs = []
            try:
                # We use .stream() instead of .invoke()
                for chunk in chain.stream({"input": prompt}):
                    # Capture context docs (usually arrives in the first chunk)
                    if "context" in chunk:
                        context_docs = chunk["context"]
                    
                    # Capture and stream answer tokens
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
                        # Update the placeholder in real-time
                        response_placeholder.markdown(full_answer + "▌")
                
                cleaned_answer = clean_latex(full_answer)
                response_placeholder.markdown(cleaned_answer)
                    
                # Format sources
                sources = []
                for doc in context_docs:
                    sources.append({
                        'file': Path(doc.metadata.get('source', 'Unknown')).name,
                        'page': doc.metadata.get('page', 'N/A'),
                        'preview': doc.page_content[:200] + "...",
                        'source_type': doc.metadata.get('source_type', 'base')
                    })
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            is_uploaded = source.get('source_type') == 'uploaded'
                            icon = "📤" if is_uploaded else "📄"
                            badge = " **(uploaded)**" if is_uploaded else ""
                            
                            st.markdown(f"**{i}.** {icon} {source['file']}{badge} (page {source['page']})")
                            st.caption(source['preview'])
                            if i < len(sources):
                                st.markdown("---")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cleaned_answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())

# Example queries (only show if no chat history and not using temp docs)
if not st.session_state.messages and not st.session_state.using_temp_docs:
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    
    col1, col2, col3 = st.columns(3)
    
    examples = [
        "What is quantum entanglement?",
        "Explain the Schrödinger equation",
        "What are Lagrangian mechanics?",
    ]
    
    for col, example in zip([col1, col2, col3], examples):
        with col:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()