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

