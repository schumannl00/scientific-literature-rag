# src/ingest.py - Optimized with parallel processing for large PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm
import pytesseract  # type: ignore
from pdf2image import convert_from_path  # type: ignore
import time
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_with_ocr(pdf_path: str | Path) -> List[Document]:
    
    try:
        start_time = time.time()
        pdf_name = Path(pdf_path).name
        logger.info(f"🔍 Starting OCR for: {pdf_name}")
        
        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=200)
        total_pages = len(images)
        logger.info(f"   Converted to {total_pages} images")

        documents: List[Document] = []
        
        for i, image in enumerate(images):
            custom_config = r'--oem 3 --psm 3' # this should handle math better
            text: str = pytesseract.image_to_string(image, config=custom_config)
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": i + 1,
                    "ocr": True,
                    "total_pages": total_pages
                }
            )
            documents.append(doc)
            
            if (i + 1) % 20 == 0: 
                elapsed: float = time.time() - start_time
                rate: float = (i + 1) / elapsed
                remaining: float = (total_pages - (i + 1)) / rate if rate > 0 else 0
                logger.info(
                    f"   Progress: {i+1}/{total_pages} pages "
                    f"({rate:.1f} pages/sec, ~{remaining/60:.1f} min remaining)"
                )
        
        elapsed = time.time() - start_time
        logger.info(
            f"✓ OCR completed in {elapsed/60:.1f} minutes "
            f"({len(images)/elapsed:.1f} pages/sec)"
        )
        return documents
    
    except Exception as e:
        logger.error(f"✗ OCR failed for {pdf_path}: {e}")
        return []


def process_single_pdf(pdf_path: str | Path) -> List[Document]:
    
    try:
        start_time = time.time()
        pdf_name = Path(pdf_path).name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 Processing: {pdf_name}")
        
        # Try fast text extraction first
        loader = PyPDFLoader(str(pdf_path))
        documents: List[Document] = loader.load()
        
        # Check if extraction worked
        total_text = "".join([doc.page_content for doc in documents])
        
        if len(total_text.strip()) < 100:  # Likely a scanned PDF
            logger.info(f"   Detected scanned PDF ({len(documents)} pages)")
            documents = extract_text_with_ocr(pdf_path)
            
            if not documents:
                logger.warning(f"   Could not extract text from {pdf_path}")
                return []
        else:
            logger.info(f"   Extracted text from {len(documents)} pages (no OCR needed)")
        
        elapsed: float = time.time() - start_time
        logger.info(f"✓ Completed {pdf_name} in {elapsed:.1f}s ({len(documents)} pages)")
        return documents
        
    except Exception as e:
        logger.error(f"✗ Failed to process {pdf_path}: {e}")
        return []


def process_pdf_batch_parallel(pdf_paths: List[Path], num_workers: int) -> List[Document]:
   
    logger.info(f"Processing {len(pdf_paths)} PDFs in parallel with {num_workers} workers...")
    
    all_documents: List[Document] = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_pdf, str(pdf_path)): pdf_path 
            for pdf_path in pdf_paths
        }
        
        # Process results as they complete with progress bar
        from concurrent.futures import as_completed
        
        with tqdm(total=len(pdf_paths), desc="Processing PDFs") as pbar:
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    pbar.update(1)
    
    return all_documents


def build_vectorstore(
    pdf_dir: str | Path = "data/pdfs",
    persist_dir: str | Path = "data/chroma_db",
    size_threshold_mb: float = 10.0,  # PDFs larger than this go to parallel processing
    max_workers: int | None = None
) -> Chroma:

    start_time = time.time()
    
    # Determine number of workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    # Find all PDFs and get their sizes
    pdf_paths: List[Path] = sorted(list(Path(pdf_dir).rglob("*.pdf")))
    logger.info(f"\n{'='*60}")
    logger.info(f"Found {len(pdf_paths)} PDF files")
    logger.info(f"{'='*60}\n")
    
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    # Sort by size and categorize
    pdf_sizes: List[Tuple[Path, float]] = [
        (p, p.stat().st_size / 1024 / 1024)  # Size in MB
        for p in pdf_paths
    ]
    pdf_sizes.sort(key=lambda x: x[1], reverse=True)  # Largest first
    
    # Split into large and small PDFs
    threshold_bytes = size_threshold_mb * 1024 * 1024
    large_pdfs = [p for p, size_mb in pdf_sizes if p.stat().st_size > threshold_bytes]
    small_pdfs = [p for p, size_mb in pdf_sizes if p.stat().st_size <= threshold_bytes]
    
    total_size_mb = sum(size for _, size in pdf_sizes)
    large_size_mb = sum(p.stat().st_size / 1024 / 1024 for p in large_pdfs)
    small_size_mb = sum(p.stat().st_size / 1024 / 1024 for p in small_pdfs)
    
    logger.info(f"Total size: {total_size_mb:.1f} MB")
    logger.info(f"Large PDFs (>{size_threshold_mb}MB): {len(large_pdfs)} files ({large_size_mb:.1f} MB)")
    logger.info(f"Small PDFs (<={size_threshold_mb}MB): {len(small_pdfs)} files ({small_size_mb:.1f} MB)\n")
    
    # Show largest files
    if large_pdfs:
        logger.info("Largest files (will be processed in parallel):")
        for i, (path, size_mb) in enumerate(pdf_sizes[:min(5, len(large_pdfs))], 1):
            logger.info(f"  {i}. {path.name} - {size_mb:.1f} MB")
        logger.info("")
    
    all_documents: List[Document] = []
    
    # Process large PDFs in parallel
    if large_pdfs:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {len(large_pdfs)} large PDFs in parallel ({max_workers} workers)")
        logger.info(f"{'='*60}\n")
        
        large_docs = process_pdf_batch_parallel(large_pdfs, max_workers)
        all_documents.extend(large_docs)
        
        logger.info(f"\n✓ Extracted {len(large_docs)} pages from large PDFs\n")
    
    # Process small PDFs sequentially (more efficient for small files)
    if small_pdfs:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {len(small_pdfs)} small PDFs sequentially")
        logger.info(f"{'='*60}\n")
        
        for i, pdf_path in enumerate(small_pdfs, 1):
            logger.info(f"[{i}/{len(small_pdfs)}]")
            docs = process_single_pdf(str(pdf_path))
            all_documents.extend(docs)
        
        logger.info(f"\n✓ Extracted pages from small PDFs\n")
    
    if not all_documents:
        raise ValueError("No documents were successfully processed!")
    
    processing_time: float = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ All PDFs processed in {processing_time/60:.1f} minutes")
    logger.info(f"Total documents (pages): {len(all_documents)}")
    logger.info(f"{'='*60}\n")
    
    # Chunking
    logger.info("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks: List[Document] = text_splitter.split_documents(all_documents)
    avg_chunk_length: float = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    logger.info(f"Created {len(chunks)} chunks (avg length: {avg_chunk_length:.0f} chars)\n")
    
    # Embeddings
    logger.info("Loading embedding model (BGE-small)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Vector store
    logger.info("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    
    total_time: float = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ COMPLETE! Total time: {total_time/60:.1f} minutes")
    logger.info(f"Vector store: {vectorstore._collection.count()} vectors")
    logger.info(f"Saved to: {persist_dir}")
    logger.info(f"{'='*60}\n")
    
    # Save metadata
    save_index_metadata(vectorstore, persist_dir)
    
    return vectorstore


def save_index_metadata(vectorstore: Chroma, persist_dir: str | Path) -> None:
    """Save metadata about indexed documents"""
    all_docs = vectorstore.get(include=['metadatas'])
    
    sources: List[str] = []
    if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
        metadatas_list = all_docs['metadatas']
        for meta in metadatas_list:
            if meta and isinstance(meta, dict):
                source = meta.get('source')
                if source and isinstance(source, str):
                    sources.append(source)
    
    # Get unique sources
    unique_sources = sorted(list(set(sources)))
    
    metadata = {
        "total_vectors": vectorstore._collection.count(),
        "unique_sources": len(unique_sources),
        "sources": unique_sources,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = Path(persist_dir) / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    # You can customize these parameters
    build_vectorstore(
        pdf_dir="data/pdfs",
        persist_dir="data/chroma_db",
        size_threshold_mb=10.0,  # Adjust based on your PDFs
        max_workers=4  # Adjust based on your CPU cores
    )





