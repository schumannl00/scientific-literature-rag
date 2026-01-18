# src/ingest_incremental.py - Add new PDFs to existing vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Set
import logging
import time
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_pdfs_to_vectorstore(
    new_pdf_dir: str | Path = "data/pdfs_new",
    persist_dir: str | Path = "data/chroma_db"
) -> Chroma:
    """
    Add new PDFs to existing vector store without re-processing old ones.
    
    Args:
        new_pdf_dir: Directory with new PDFs to add
        persist_dir: Existing vector store directory
        
    Returns:
        Updated Chroma vector store
    """
    # Load existing vector store
    logger.info(f"Loading existing vector store from {persist_dir}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )
    
    initial_count = vectorstore._collection.count()
    logger.info(f"Current vector count: {initial_count:,}")
    
    # Get list of sources already in the vector store
    existing_sources: Set[str] = set()
    all_docs = vectorstore.get(include=['metadatas'])
    
    if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
        metadatas_list = all_docs['metadatas']
        for meta in metadatas_list:
            # Safely extract source from metadata
            if meta and isinstance(meta, dict):
                source = meta.get('source')
                if source and isinstance(source, str):
                    existing_sources.add(source)
    
    logger.info(f"Already indexed: {len(existing_sources)} unique sources")
    
    # Process new PDFs
    new_pdf_paths = list(Path(new_pdf_dir).rglob("*.pdf"))
    logger.info(f"Found {len(new_pdf_paths)} PDFs in {new_pdf_dir}")
    
    # Filter out already-processed PDFs
    pdfs_to_process = [
        p for p in new_pdf_paths
        if str(p) not in existing_sources
    ]
    
    if not pdfs_to_process:
        logger.info("No new PDFs to process!")
        return vectorstore
    
    logger.info(f"Processing {len(pdfs_to_process)} new PDFs...")
    logger.info(f"Skipping {len(new_pdf_paths) - len(pdfs_to_process)} already indexed PDFs")
    
    # Process new PDFs (reuse your existing process_single_pdf function)
    from src.ingest import process_single_pdf
    
    all_documents: List[Document] = []
    for i, pdf_path in enumerate(pdfs_to_process, 1):
        logger.info(f"[{i}/{len(pdfs_to_process)}]")
        docs = process_single_pdf(str(pdf_path))
        all_documents.extend(docs)
    
    if not all_documents:
        logger.warning("No new documents extracted!")
        return vectorstore
    
    logger.info(f"Extracted {len(all_documents)} pages from new PDFs")
    
    # Chunk new documents
    logger.info("Chunking new documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents(all_documents)
    logger.info(f"Created {len(chunks)} new chunks")
    
    # Add to existing vector store
    logger.info("Embedding and adding new chunks to vector store...")
    start_time = time.time()
    vectorstore.add_documents(chunks)
    elapsed = time.time() - start_time
    
    final_count = vectorstore._collection.count()
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Added {final_count - initial_count:,} new vectors in {elapsed/60:.1f} minutes")
    logger.info(f"Total vectors: {final_count:,}")
    logger.info(f"{'='*60}\n")
    
    return vectorstore


def list_indexed_sources(persist_dir: str | Path = "data/chroma_db") -> List[str]:
    """
    List all sources currently in the vector store.
    
    Args:
        persist_dir: Vector store directory
        
    Returns:
        Sorted list of unique source paths
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
    )
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )
    
    all_docs = vectorstore.get(include=['metadatas'])
    
    sources: List[str] = []
    if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
        for meta in all_docs['metadatas']:
            if meta and isinstance(meta, dict):
                source = meta.get('source')
                if source and isinstance(source, str):
                    sources.append(source)
    
    unique_sources = sorted(list(set(sources)))
    
    print(f"\nIndexed sources ({len(unique_sources)} unique):")
    print("=" * 60)
    for i, source in enumerate(unique_sources, 1):
        print(f"{i}. {Path(source).name}")
    
    return unique_sources


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # List indexed sources
        list_indexed_sources()
    else:
        
        add_pdfs_to_vectorstore(
            new_pdf_dir="data/new_pdfs",
            persist_dir="data/chroma_db"
        )