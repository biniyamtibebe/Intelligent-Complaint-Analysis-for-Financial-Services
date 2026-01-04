# Intelligent Complaint Analysis for Financial Services  
**RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights**

This repository contains the work completed for an internal AI tool at **CrediTrust Financial**, designed to help Product Managers, Customer Support, Compliance, and Executive teams quickly understand customer pain points from complaint narratives across four key products:

- Credit Cards  
- Personal Loans  
- Savings Accounts  
- Money Transfers  

The tool uses **Retrieval-Augmented Generation (RAG)** to enable natural-language querying of thousands of real customer complaints (sourced from the CFPB Consumer Complaint Database).

## Project Status (as of Interim Submission – 04 Jan 2026)

**Completed Tasks:**
- Task 1: Exploratory Data Analysis and Data Preprocessing  
- Task 2: Text Chunking, Embedding, and Vector Store Indexing (on stratified sample)

**In Progress / Upcoming:**
- Task 3: Full RAG Pipeline with LLM integration (using pre-built embeddings on full dataset)
- Task 4: Gradio/Streamlit Chatbot UI

## Project Structure
'''rag-complaint-chatbot/
├── data/
│   ├── raw/                    # Original CFPB complaints.csv (not committed)
│   └── processed/
│       └── filtered_complaints.csv   # Cleaned & filtered dataset (Task 1 output)
├── vector_store/
│   └── chroma_sample_index/    # Persisted ChromaDB index from Task 2 (sample data)
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb (or .py)
│   └── task2_chunking_embedding_indexing.py
├── src/                        # Future: modular code for RAG pipeline
├── app.py                      # Future: UI entrypoint
├── requirements.txt
├── README.md                   # This file
└── .gitignore 
'''

---

## Task 1: Exploratory Data Analysis and Preprocessing

### Objective
Understand the raw CFPB dataset, filter to relevant products, clean narratives, and prepare high-quality text for embedding.

### Key Steps Performed
- Loaded the full CFPB Consumer Complaint Database (~4M+ rows).
- Analyzed product distribution, narrative presence, and word count distribution.
- Filtered to only four target product categories (Credit Cards, Personal Loans, Savings Accounts, Money Transfers).
- Removed complaints without narratives and very short narratives (<10 words).
- Cleaned narratives:
  - Lowercased text
  - Removed special characters and extra whitespace
  - Stripped common boilerplate phrases
- Added standardized `product_category` column aligned with CrediTrust products.

### Output
- `data/processed/filtered_complaints.csv`: Cleaned dataset ready for chunking and embedding.
- EDA visualizations and summary insights included in notebook.

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objective
Convert long complaint narratives into searchable chunks and build a semantic search index on a representative sample.

### Sampling Strategy
- Created a **stratified sample of 12,000 complaints** from the filtered dataset.
- Stratification by `product_category` ensures proportional representation of all four products.
- This size balances diversity, computational feasibility, and realistic testing during development.

### Chunking Strategy
- Used `langchain.text_splitters.RecursiveCharacterTextSplitter`
- Parameters:
  - `chunk_size = 500` characters (~100–120 words)
  - `chunk_overlap = 50` characters
- Rationale: Small chunks improve retrieval precision while overlap preserves context across boundaries. Size fits well within the effective context window of the embedding model.

### Embedding Model
- **sentence-transformers/all-MiniLM-L6-v2**
- Reasons for selection:
  - Produces compact 384-dimensional vectors
  - Excellent performance on semantic similarity tasks (top-ranked on MTEB)
  - Fast inference on CPU, low memory footprint (~80MB)
  - Matches the model used in the pre-built full-dataset embeddings (ensures consistency later)

### Vector Store
- **ChromaDB** with persistent storage
- Collection: `complaint_chunks_sample`
- Distance metric: Cosine similarity
- Rich metadata stored per chunk:
  - `complaint_id`, `product_category`, `product`, `date_received`
  - `chunk_index`, `total_chunks`, `source_narrative_length`

### Output
- Persisted vector database at `vector_store/chroma_sample_index/`
- Contains ~35,000–50,000 chunks (depending on average narrative length)
- Fully queryable for testing retrieval performance

### Test Query Example
```python
collection.query(
    query_texts=["unfair late fees on credit card"],
    n_results=5
)