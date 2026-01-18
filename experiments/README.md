#  RAG Experiments & Decision Log

This directory tracks the evolution of the Physics-RAG system. Below are the key technical decisions made to handle the complexity of mathematical research papers.

---

##  Implemented Architectures

### 1. Maximal Marginal Relevance (MMR)
* **Problem:** Standard similarity search often retrieved 4-5 near-identical chunks from the same paper, leading to "information blindness."
* **Solution:** Switched to `search_type="mmr"`. 
* **Result:** Significantly improved diversity. The system now pulls from multiple papers or distinct sections, providing a more holistic view of complex physics topics.

### 2. FlashRank Reranking
* **Problem:** The most similar vector (bi-encoder) isn't always the most relevant for logical physics derivations.
* **Solution:** Implemented `FlashrankRerank`. We pull 10 candidates via MMR and use a cross-encoder to re-score them.
* **Result:** Higher precision in answers. The LLM receives the most "logically" relevant context, reducing hallucinations in formulas.

### 3. Ephemeral In-Memory Scaling (Split DB)
* **Problem:** Uploading a new PDF was triggering a full re-index of the 180+ base papers, causing near 1h hangs on CPU.
* **Solution:** * **Base Store:** Persistent ChromaDB on disk.
    * **Temp Store:** `chromadb.EphemeralClient()` for user uploads (RAM-only).
    * **Logic:** Used `MergerRetriever` to query both simultaneously without re-embedding the base library.
* **Result:** Document processing time dropped from one hour to a few seconds for papers and 1-2 minutes for 400+ page books.

### 4. Dynamic LaTeX Rendering
* **Problem:** Qwen 2.5 often outputted `\[ ... \]` or `\( ... \)` which broke the Streamlit KaTeX renderer.
* **Solution:** * **Regex Sanitization:** A `clean_latex` helper converts all brackets to `$$` and `$`.
    * **Session State Persistence:** Answers are cleaned *before* being stored in history to prevent formatting breaks on reruns.
* **Result:** Seamless mathematical notation for tensors, operators, and field equations.

---

##  Future Observations
* [ ] **Chunk Size:** Currently 1000/200. Need to test if smaller chunks improve "needle-in-a-haystack" retrieval for specific constants.
* [ ] **Model Scaling:** Compare Qwen 2.5 7B vs 1.5B for latency-to-accuracy trade-off.
* [ ] **Hard Questions:** Need to move away from definitions to "cross-paper reasoning."
