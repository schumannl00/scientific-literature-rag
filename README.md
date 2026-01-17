# Scientific RAG Pipeline (Work in Progress)

This repository contains a work-in-progress retrieval-augmented generation (RAG) system built over a curated corpus of scientific literature, lecture notes, and scanned PDF documents from my Master’s studies.

The goal of this project is to design a **domain-specific, local RAG pipeline** that enables grounded, accurate question answering over technically complex documents. The system emphasizes modularity, transparent retrieval, and reliable document preprocessing, and is intended as a transferable foundation for sensitive domains such as medical or clinical text corpora.

---

## Motivation

Large language models are powerful but unreliable when used without access to trusted sources. In scientific and technical domains, accuracy, traceability, and controlled data handling are essential.

This project explores:
- ingestion and cleaning of heterogeneous scientific documents (including scanned PDFs, which are common, especially for older books),
- OCR-based preprocessing pipelines using Tesseract,
- embedding and retrieval workflows,
- and modular integration of local language models for grounded generation.

---
Planned Architecture 

The system is built around **LangChain** for orchestration and **local models via Ollama**, enabling full local execution without dependence on external APIs.

---

## Key Features (in progress)

- Scientific PDF ingestion (digital and scanned)
- OCR pipelines for scanned sources
- Document cleaning and metadata extraction
- Chunking and embedding workflows
- Vector-based retrieval
- LangChain-based RAG orchestration
- Local LLM inference using Ollama
- Grounded answering with source attribution

---

## Tech Stack

- Python  
- LangChain  
- Ollama (local LLM inference)  
- FAISS / Chroma (vector storage, planned)  
- OCR (e.g. Tesseract, planned)  

---

## Current Status

This project is under active development.  
Initial focus is on building robust ingestion and preprocessing pipelines, followed by retrieval workflows and evaluation of answer grounding and accuracy.

---

## Roadmap

- [ ] PDF loaders and OCR preprocessing
- [ ] Text cleaning and chunking strategies
- [ ] Embedding and vector store integration
- [ ] LangChain RAG pipeline
- [ ] Local model benchmarking
- [ ] Simple evaluation suite for answer grounding
- [ ] Documentation and example notebooks

---

## Disclaimer

This system is intended for research and experimentation.  
It is not a production system and is continuously evolving.
