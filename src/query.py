# query.py - Modernized Physics RAG (v1.2.x)
from langchain_core.documents import Document                # New standard for schema
from langchain_core.prompts import ChatPromptTemplate         # Better for ChatModels
from langchain_huggingface import HuggingFaceEmbeddings        # New standard for HF
from langchain_chroma import Chroma                            # Dedicated partner package
from langchain_ollama import OllamaLLM                        # Updated Ollama integration
from langchain_classic.chains import create_retrieval_chain            # Modern RAG orchestration
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_rag_chain(
    persist_dir: str = "data/chroma_db",
    model: str = "qwen2.5:7b",
    temperature: float = 0.1,
    k: int = 4
):
    """
    Builds a modern RAG chain using LCEL.
    Documentation:
    - Chroma: Dedicated package for vector storage
    - create_retrieval_chain: Links the retriever to the LLM response
    """
    
    # 1. Load Embeddings (Must match ingest.py: BGE-Base for math accuracy/speed)
    logger.info("Loading embedding model (BGE-Base)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Load Vector Store
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )
    
    # 3. Initialize LLM (Ollama v2 integration)
    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        num_ctx=3072)
    
    # 4. Define the Prompt (System prompt + Context + User Input)
    system_prompt = (
        "You are a helpful physics research assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use professional physics terminology and precise math symbols.\n\n"
        "If the question is not related to physics research, politely respond that you are specialized in physics research only.\n\n"
        "if you are unsure about the answer, prompt the user to refine their question or provide more context e.g. the full name of the author or paper.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Create the Chains
    # combine_docs_chain: Handles how context is 'stuffed' into the prompt
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # retriever: Search logic
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # retrieval_chain: The final LCEL object
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    logger.info("Modern RAG chain ready.")
    return rag_chain

def query(question: str, chain: Any) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\nAnswer: ", end="", flush=True)
    
    full_result = {}
    
    
    # This yields chunks as the CPU processes them
    for chunk in chain.stream({"input": question}):
       
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            
            full_result["answer"] = full_result.get("answer", "") + chunk["answer"]
            
        
        if "context" in chunk:
            full_result["context"] = chunk["context"]

    print(f"\n\n{'='*60}")
    print("Sources (Validation):")
    print(f"{'='*60}")
    
    # Validate sources using the accumulated 'context'
    source_documents = full_result.get("context", [])
    for i, doc in enumerate(source_documents, 1):
        source_path = doc.metadata.get('source', 'Unknown')
        source_name = Path(source_path).name
        page = doc.metadata.get('page', 'N/A')
        
        print(f"{i}. {source_name} (pg {page})")
        print(f"   Snippet: {doc.page_content[:100].strip()}...")
        
    return full_result

if __name__ == "__main__":
    # Ensure you have the updated requirements installed!
    rag_chain = load_rag_chain()
    
    test_questions = [
        "What did Ko Sanders contribute to the theory of the locally covariant Dirac field ?",
        "Explain how the operator product expansion converges in the massless case"
    ]
    
    for q in test_questions:
        query(q, rag_chain)